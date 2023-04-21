```python
#!pip install peft
#!pip install transformers==4.28.1
#!pip install accelerate
#!pip install loralib
#!pip install evaluate
#!pip install tqdm
#!pip install datasets

#!wget https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv

import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import peft
peft.__version__

data_file = "./ChnSentiCorp_htl_all.csv" # 数据文件路径，数据需要提前下载
# 加载数据集
dataset = load_dataset("csv", data_files=data_file)
dataset = dataset.filter(lambda x: x["review"] is not None)
datasets = dataset["train"].train_test_split(0.2, seed=123)

model_name_or_path = "hfl/chinese-roberta-wwm-ext"

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

max_length = 86

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def process_function(examples):
  tokenized_examples = tokenizer(examples["review"], truncation=True, max_length=max_length)
  tokenized_examples["labels"] = examples["label"]
  return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = predictions.argmax(axis=-1)
  return accuracy_metric.compute(predictions=predictions, references=labels)


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
batch_size = 64
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

# 训练器配置
p_type = "prefix-tuning"
if p_type == "prefix-tuning":
  peft_type = PeftType.PREFIX_TUNING
  peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
elif p_type == "prompt-tuning":
  peft_type = PeftType.PROMPT_TUNING
  peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
elif p_type == "p-tuning":
  peft_type = PeftType.P_TUNING
  peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
elif p_type == "lora":
  peft_type = PeftType.LORA
  peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
# print(peft_type)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
if p_type is not None:
  model = get_peft_model(model, peft_config)
  model.print_trainable_parameters()
else:
  def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

  print_trainable_parameters(model)

lr = 3e-4
num_epochs = 3
optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

device = "cuda"
model.to(device)
metric = evaluate.load("accuracy")
save_dir = "./prefix-tuning"
import time
start = time.time()
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    total_loss = 0.
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch} loss {total_loss}:", eval_metric)
    model.save_pretrained(save_dir)
end = time.time()

print("耗时：{}分钟".format((end-start) / 60))


from peft import PeftModel
from transformers import pipeline
model_name_or_path = "hfl/chinese-roberta-wwm-ext"
lora_model_dir = "./prefix-tuning"
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
# model.cuda()
# model.eval()
lora_model = PeftModel.from_pretrained(model, lora_model_dir, torch_dtype=torch.float32)
lora_model.cuda()
lora_model.eval()
text = '叫酒店早上6点50叫醒，但是在7点10还没有人叫，差点误了飞机！'
with torch.no_grad():
  inputs = tokenizer(text, truncation=True, max_length=86, padding="max_length", return_tensors="pt")
  for k, v in inputs.items():
    inputs[k] = v.cuda()
  # print(inputs)
  output = lora_model(**inputs)
  logits = output.logits
  print(logits)
```

