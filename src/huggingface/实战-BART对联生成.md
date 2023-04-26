````python
!pip install peft
!pip install transformers==4.28.1
!pip install accelerate
!pip install loralib
!pip install evaluate
!pip install tqdm
!pip install datasets
!pip install trl
!pip install wandb
!pip install lawrouge
!git clone https://www.modelscope.cn/datasets/minisnow/couplet_samll.git
````

```python
import json
import pandas as pd
import numpy as np
import lawrouge
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline, pipeline
from datasets import load_dataset, Dataset


# ============================= 
# 加载数据
train_path = "couplet_samll/train.csv"
train_dataset = Dataset.from_csv(train_path)
test_path = "couplet_samll/test.csv"
test_dataset = Dataset.from_csv(test_path)

# 转换为模型需要的格式
def tokenize_dataset(tokenizer, dataset, max_len):
  def convert_to_features(batch):
    text1 = batch["text1"]
    text2 = batch["text2"]
    inputs = tokenizer.batch_encode_plus(
      text1,
      max_length=max_len,
      padding="max_length",
      truncation=True,
    )
    targets = tokenizer.batch_encode_plus(
      text2,
      max_length=max_len,
      padding="max_length",
      truncation=True,
    )
    outputs = {
      "input_ids": inputs["input_ids"],
      "attention_mask": inputs["attention_mask"],
      "target_ids": targets["input_ids"],
      "target_attention_mask": targets["attention_mask"]
    }
    return outputs
  
  dataset = dataset.map(convert_to_features, batched=True)
  # Set the tensor type and the columns which the dataset should return
  columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
  dataset.with_format(type='torch', columns=columns)
  dataset = dataset.rename_column('target_ids', 'labels')
  dataset = dataset.rename_column('target_attention_mask', 'decoder_attention_mask')
  dataset = dataset.remove_columns(['text1', 'text2'])
  return dataset

tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
max_len = 24
train_data = tokenize_dataset(tokenizer, train_dataset, max_len)
test_data = tokenize_dataset(tokenizer, test_dataset, max_len)

from transformers import default_data_collator
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader

train_batch_size = 64
eval_batch_size = 64

train_dataset = train_data
train_dataloader = DataLoader(
    train_dataset, collate_fn=default_data_collator, shuffle=True, batch_size=train_batch_size, pin_memory=True
)
test_dataset = test_data
test_dataloader = DataLoader(
    test_dataset, collate_fn=default_data_collator, batch_size=eval_batch_size, pin_memory=True
)

# optimizer
model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
lr = 3e-4
num_epochs = 1
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
    t = tqdm(train_dataloader)
    for step, batch in enumerate(t):
        for k, v in batch.items():
            batch[k] = v.cuda()
        outputs = model(**batch)
        loss = outputs.loss
        t.set_description("loss：{:.6f}".format(loss.item()))
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    train_epoch_loss = total_loss / len(train_dataloader)
    model.save_pretrained("bart-couplet/")
    tokenizer.save_pretrained("bart-couplet/")
    print(f"epoch:{epoch+1}/{num_epochs} loss:{train_epoch_loss}")
```

预测：

```python
from transformers import Text2TextGenerationPipeline
model_path = "bart-couplet"
# model_path = "fnlp/bart-base-chinese"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
generator = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

max_len = 24

test_path = "couplet_samll/test.csv"
test_data = pd.read_csv(test_path)
texts = test_data["text1"].values.tolist()
labels = test_data["text2"].values.tolist()

results = generator(texts, max_length=max_len, eos_token_id=0, pad_token_id=0, do_sample=True)
for text, label, res in zip(texts, labels, results):
  print(res)
  print("上联：", text)
  print("真实下联：", label)
  print("预测下联：", "".join(res["generated_text"].split(" ")))
  print("="*100)
```

```python
model = BartForConditionalGeneration.from_pretrained(model_path)
model = model.to("cuda")
model.eval()
inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)
# 生成
outputs = model.generate(input_ids, 
              attention_mask=attention_mask, 
              max_length=max_len, 
              do_sample=True, 
              pad_token_id=0,
              eos_token_id=102)
# 将token转换为文字
output_str = tokenizer.batch_decode(outputs, skip_special_tokens=False)
output_str = [s.replace(" ","") for s in output_str]
for text, label, pred in zip(texts, labels, output_str):
  print("上联：", text)
  print("真实下联：", label)
  print("预测下联：", pred)
  print("="*100)
```

