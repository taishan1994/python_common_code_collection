# 安转依赖

```python
python3.9
torch==2.0+cu17
tokenizers==0.13.3

!pip install peft==0.2.0
!pip install transformers==4.28.1
!pip install accelerate==0.18.0
!pip install loralib
!pip install evaluate==0.4.0
!pip install tqdm
!pip install datasets==2.11.0
!pip install deepspeed==0.9.1
!pip install mpi4py
!pip install trl==0.4.1
!wget https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv
```

# 训练GPT2生成评论数据

```python
from datasets import load_dataset
data_file = "./ChnSentiCorp_htl_all.csv" # 数据文件路径，数据需要提前下载
# 加载数据集
dataset = load_dataset("csv", data_files=data_file)
dataset = dataset.filter(lambda x: x["review"] is not None)
dataset = dataset["train"].train_test_split(0.2, seed=123)

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from pprint import pprint
model_name_or_path = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# example = {'label': 1, 'review': '早餐太差，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。房间本身很好。'}

max_length = 86

def process(example):
  text = example["review"]
  # text = ["399真的很值得之前也住过别的差不多价位的酒店式公寓没有这间好厨房很像厨房很大整个格局也都很舒服早上的早餐我订的8点半的已经冷了。。。位置啊什么还是很好的下次还会去服务也很周到"]
  batch_size = len(text)
  inputs = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)
  inputs["labels"] = []
  for i in range(batch_size):
    input_ids = inputs["input_ids"][i]
    if len(input_ids) + 1 <= max_length:
      inputs["input_ids"][i] = input_ids + [tokenizer.pad_token_id] + [0] * (max_length - len(input_ids) - 1)
      inputs["labels"].append(input_ids + [tokenizer.pad_token_id] + [-100] * (max_length - len(input_ids) - 1))
      inputs["attention_mask"][i] = [1] * len(input_ids) + [0] + [0] * (max_length - len(input_ids) - 1)
    else:
      inputs["input_ids"][i] = input_ids[:max_length-1] + [tokenizer.pad_token_id]
      inputs["labels"].append(inputs["input_ids"][i])
      inputs["attention_mask"][i] = [1] * max_length

    inputs["token_type_ids"][i] = [0] * max_length
    # for k, v in inputs.items():
    #   print(k, len(v[0]))
    # assert len(inputs["labels"][i]) == len(inputs["input_ids"][i]) == len(inputs["token_type_ids"][i]) == len(inputs["attention_mask"][i]) == 86
  return inputs

# process(None)

train_dataset = dataset["train"].map(process, batched=True, num_proc=1, remove_columns=dataset["train"].column_names)
test_dataset = dataset["test"].map(process, batched=True, num_proc=1, remove_columns=dataset["test"].column_names)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.utils.data import DataLoader
train_batch_size = 64
eval_batch_size = 64

train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, shuffle=True, batch_size=train_batch_size, pin_memory=True
    )

test_dataloader = DataLoader(
    test_dataset, collate_fn=default_data_collator, batch_size=eval_batch_size, pin_memory=True
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
import torch
# optimizer
num_epochs = 10
lr = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# lr scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.cuda()

from tqdm import tqdm
for epoch in range(num_epochs):
  model.train()
  total_loss = 0
  for step, batch in enumerate(t:=tqdm(train_dataloader)):
    for k,v in batch.items():
      batch[k] = v.cuda()
    outputs = model(
        input_ids=batch["input_ids"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = outputs.loss
    t.set_description("loss：{:.6f}".format(loss.item()))
    total_loss += loss.detach().float()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
  
  train_epoch_loss = total_loss / len(train_dataloader)
  model.save_pretrained("gpt2-chinese/")
  print(f"epoch:{epoch}/{num_epochs} loss:{train_epoch_loss}")
```

# 训练奖励模型-情感分析

```python
#pip install peft
#pip install transformers==4.28.1
#pip install accelerate
#pip install loralib
#pip install evaluate
#pip install tqdm
#pip install datasets

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
p_type = "lora"
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
save_dir = p_type if p_type is not None else "bert"
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
```

# 利用PPO进行可控文本生成

```python
seed = 123
np.random.seed(seed)

config = PPOConfig(
    model_name="./gpt2-chinese", # 使用训练好的模型
    steps=51200, 
    learning_rate=1.41e-5, 
    remove_unused_columns=False, 
    # log_with="wandb"
)

txt_in_len = 5
txt_out_len = 20


# ==================================
# 定义模型
gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
gpt2_model_ref = create_reference_model(gpt2_model)
# 使用原来模型的tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

gpt2_tokenizer.eos_token = gpt2_tokenizer.pad_token
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
# ==================================

# ==================================
# 加载数据
data_file = "./ChnSentiCorp_htl_all.csv"
dataset = load_dataset("csv", data_files=data_file)
dataset = dataset.filter(lambda x: x["review"] is not None)
dataset = dataset["train"]
print(dataset)

def tokenize(sample):
  sample["input_ids"] = gpt2_tokenizer.encode(sample["review"], add_special_tokens=False)[:txt_in_len]
  sample["query"] = "".join(gpt2_tokenizer.decode(sample["input_ids"]).split(" "))
  return sample

dataset = dataset.map(tokenize, batched=False)
print(dataset)
# 将指定的列名转换为torch的格式
dataset.set_format(type="torch", columns=["input_ids", "label"], output_all_columns=True)

# ==================================


# ==================================
# PPO主代码
def collator(data):
  # 构建batch数据
  return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(
    config, 
    gpt2_model, 
    gpt2_model_ref, 
    gpt2_tokenizer, 
    dataset, 
    data_collator=collator)

if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
else:
    device = ppo_trainer.accelerator.device

# 构建reward模型
sentiment_pipe = pipeline(
    "sentiment-analysis", 
    "./roberta-chinese", 
    tokenizer=gpt2_tokenizer,
    device=device)

# 提取出正面的分数
def extract_pipe_output(outputs):
  positive_logits = []
  for out in outputs:
    for element in out:
      if element["label"] == "LABEL_1":
        positive_logits.append(torch.tensor(element["score"]))
  return positive_logits

# 加入prompt
ctrl_str = ["正面：", "负面："]
ctrl_tokens = dict((s, gpt2_tokenizer.encode(s, add_special_tokens=False, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)
ctrl_tokens

# 定义生成模型参数
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
    "max_new_tokens": txt_out_len,
    "eos_token_id": gpt2_tokenizer.eos_token_id,
}


# 定义奖励模型参数
sentiment_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

def pos_logit_to_reward(logit, task):
    """如果prompt是正面，则奖励为正，否则，奖励为负
    """
    for i in range(len(logit)):
        if task[i] == "负面：":
            logit[i] = -logit[i]   
        else:
            pass
    return logit

for epoch in range(2):
    for batch in tqdm(ppo_trainer.dataloader):
        logs, game_data, = (
            dict(),
            dict(),
        )

        #### 为每一个样本随机选一个prompt
        task_list = choices(ctrl_str, k=config.batch_size)
        game_data["query"] = [t + q for t, q in zip(task_list, batch["query"])]
        query_tensors = [torch.cat((ctrl_tokens[t], input_ids)) for t, input_ids in zip(task_list, batch["input_ids"])]

        #### 使用GPT2生成结果
        response_tensors = []
        for query in query_tensors:
            query_length = len(query)
            response = ppo_trainer.generate(query, **generation_kwargs)
            # print(gpt2_tokenizer.decode(query))
            # print(gpt2_tokenizer.decode(response.squeeze()[query_length:txt_out_len]))
            # 这里使用的模型从前往后解码
            response_tensors.append(response.squeeze()[query_length:txt_out_len])
            break
        game_data["response"] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### 使用奖励模型对输出结果进行评分
        texts = [q + "".join(r.split(" ")) for q, r in zip(batch["query"], game_data["response"])]
        print(texts[0])
        # 提取出LABEL_1(正面)对应的分数
        logits = extract_pipe_output(sentiment_pipe(texts, **sentiment_pipe_kwargs))
        rewards = pos_logit_to_reward(logits, task_list)

        #### Run PPO training
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        for cs in ctrl_str:
            key = "env/reward_" + cs.strip("[]")
            stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
        ppo_trainer.log_stats(stats, game_data, rewards)

    gpt2_model.save_pretrained(f"./ppo-chinese/epoch-{epoch}/")
    gpt2_tokenizer.save_pretrained(f"./ppo-chinese/epoch-{epoch}/")

# ==================================
```

