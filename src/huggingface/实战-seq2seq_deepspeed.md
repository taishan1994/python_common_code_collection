下载数据：!git clone https://github.com/EagleW/ACL_titles_abstracts_dataset

```
# pip install transformers==4.21.0 datasets evaluate
# pip install nltk rouge_score
```

加载数据脚本：load_script.py

```python
import datasets
from datasets import DownloadManager, DatasetInfo

logger = datasets.logging.get_logger(__name__)

_DOCUMENT = "abstract"
_SUMMARY = "title"


class AclSummarization(datasets.GeneratorBasedBuilder):

    def _info(self) -> DatasetInfo:
        """
            info方法，要定义数据集的信息
            *** 定义 feature
            涉及两个字段：_DOCUMENT和_SUMMARY，datasets.Value()声明字段的类型
        :return:
        """
        return datasets.DatasetInfo(
            description="ACL标题摘要数据集",
            features=datasets.Features({_DOCUMENT: datasets.Value("string"), _SUMMARY: datasets.Value("string")}),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """
            返回datasets.SplitGenerator
            涉及两个参数：name和gen_kwargs
            name: 指定数据集的划分
            gen_kwargs: 指定要读取的文件的路径，与_generate_examples的入参数一致
        :param dl_manager:
        :return: [ datasets.SplitGenerator ]
        """
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                        gen_kwargs={"filepath": "./acl_titles_and_abstracts.txt"})]

    def _generate_examples(self, filepath):
        """
            生成具体的样本，使用yield
            需要额外指定key，id从0开始自增就可以
        :param filepath:
        :return:
        """
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            example = {}
            for line in f.readlines():
                if line.strip() == "":
                    yield key, example
                    example = {}
                    key += 1
                else:
                    if _SUMMARY not in example:
                        example[_SUMMARY] = line.strip()
                    else:
                        example[_DOCUMENT] = line.strip()
```

解析数据：

```python
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
prompt_start = "Summarize the following news article:\n"
generation_start = "\nSummary:\n"
prompt_template = f"{prompt_start}{{input}}{generation_start}"
text_column = "abstract"
summary_column = "title"

dataset = load_dataset("./load_script.py", split="train")
dataset = dataset.train_test_split(test_size=0.2)
# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained("t5-base")

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

prompt_lenght = len(tokenizer(prompt_template.format(input=""))["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_lenght
print(f"Prompt lenght: {prompt_lenght}")
print(f"Max input lenght: {max_sample_length}")

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x[text_column], truncation=True), batched=True, remove_columns=[text_column, summary_column]
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_sample_length)
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x[summary_column], truncation=True), batched=True, remove_columns=[text_column, summary_column]
)
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# use 95th percentile as max target length
max_target_length = int(np.percentile(target_lenghts, 95))
print(f"Max target length: {max_target_length}")
```

deepspeed配置文件：ds_flan_t5_z3_config_bf16.json

```json
{
  "bf16": {
    "enabled": "auto"
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 2000,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

主运行代码：

```python
import os
import argparse
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk, load_dataset
import torch
import evaluate
import nltk
import numpy as np

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

nltk.download("punkt", quiet=True)

# Metric
metric = evaluate.load("rouge")
# evaluation generation args
gen_kwargs = {
    "early_stopping": True,
    "length_penalty": 2.0,
    "max_new_tokens": 50,
    "min_length": 30,
    "no_repeat_ngram_size": 3,
    "num_beams": 4,
}


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the already processed dataset.")
    parser.add_argument(
        "--repository_id", type=str, default=None, help="Hugging Face Repository id for uploading models"
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Path to deepspeed config file.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_known_args()
    return args



def training_function(args):
    # set seed
    set_seed(args.seed)

    # load dataset from disk and tokenizer
    # train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    # eval_dataset = load_from_disk(os.path.join(args.dataset_path, "eval"))
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

     # 这里修改加载的数据
    
    dataset = load_dataset("./load_script.py", split="train")

    dataset = dataset.train_test_split(test_size=0.2)

    prompt_start = "Summarize the following news article:\n"
    generation_start = "\nSummary:\n"
    prompt_template = f"{prompt_start}{{input}}{generation_start}"
    text_column = "abstract"
    summary_column = "title"

    # prompt_lenght = len(tokenizer(prompt_template.format(input=""))["input_ids"])
    # max_sample_length = tokenizer.model_max_length - prompt_lenght

    max_source_length = 495
    # max_target_length = 34
    max_target_length = args.generation_max_length

    def preprocess_function(sample, padding="max_length"):
        # created prompted input
        inputs = [prompt_template.format(input=item) for item in sample[text_column]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=sample[summary_column], max_length=max_target_length, padding=padding, truncation=True
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    train_dataset = dataset["train"].map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features), num_proc=4)
    test_dataset = dataset["test"].map(preprocess_function, batched=True, remove_columns=list(dataset["test"].features), num_proc=4)

    train_dataset.save_to_disk(os.path.join(args.dataset_path, "train"))
    test_dataset.save_to_disk(os.path.join(args.dataset_path, "eval"))

    # train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    # eval_dataset = load_from_disk(os.path.join(args.dataset_path, "eval"))

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(labels != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Define training args
    # output_dir = args.repository_id if args.repository_id else args.model_id.split("/")[-1]
    output_dir = args.model_id.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save our tokenizer and create model card
    tokenizer.save_pretrained(output_dir)
    trainer.create_model_card()
    # Push the results to the hub
    if args.repository_id:
        trainer.push_to_hub()


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
```

运行指令：

```python
!deepspeed --num_gpus=1 run_seq2seq.py \
    --model_id "t5-base" \
    --dataset_path "data" \
    --epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --generation_max_length 34 \
    --lr 1e-4 \
    --deepspeed ds_flan_t5_z3_config_bf16.json 
```

预测：

```python
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-efficient-tiny_check/checkpoint-2176")
# tokenizer = AutoTokenizer.from_pretrained("t5-efficient-tinym/checkpoint-2176")
tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-tiny")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
text = "learning taxonomy for technical terms is difficult and tedious task , especially when new terms should be included . the goal of this paper is to assign taxonomic relations among technical terms . we propose new approach to the problem that relies on term specificity and similarity measures . term specificity and similarity are necessary conditions for taxonomy learning , because highly specific terms tend to locate in deep levels and semantically similar terms are close to each other in taxonomy . we analyzed various features used in previous researches in view of term specificity and similarity , and applied optimal features for term specificity and similarity to our method ." 
prompt_start = "Summarize the following news article:\n"
generation_start = "\nSummary:\n"
prompt_template = f"{prompt_start}{{input}}{generation_start}"
text = prompt_template.format(input=text)
summarizer(text)
```

