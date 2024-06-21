# 实体识别接口

```python
import traceback
from transformers import AutoModel, AutoTokenizer
import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

# 初始化FastAPI应用
app = FastAPI()

# 模型路径
model_path = "model_hub/uie-base"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# 检查并使用GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

print("完成模型的加载！！")


# 定义请求和响应的数据模型
class RelationExtractionRequest(BaseModel):
    text: str
    labels: List[str]


class RelationExtractionResponse(BaseModel):
    text: str
    result: Any
    message: str
    code: int


# 辅助函数：将返回结果中的类型转换为标准Python类型
def convert_to_standard_types(data):
    if isinstance(data, dict):
        return {k: convert_to_standard_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_standard_types(i) for i in data]
    elif isinstance(data, (int, float, str)):
        return data
    elif isinstance(data, (np.int32, np.int64, np.float32, np.float64)):
        return data.item()
    else:
        return str(data)


@app.post("/extract_relation", response_model=RelationExtractionResponse)
async def extract_relation(data: RelationExtractionRequest, response: Response):
    # 将输入数据转为字符串并移动到GPU设备
    # inputs = tokenizer(data.text, return_tensors="pt").to(device)
    message = "请求成功"
    code = 200
    # 使用模型进行推理并获取输出, 这里使用原文本，不需要编码后的文本
    try:
        outputs = model.predict(tokenizer, data.text, schema=data.labels)
        outputs = outputs[0]
        standardized_outputs = convert_to_standard_types(outputs)
    except Exception as e:
        standardized_outputs = {}
        message = traceback.format_exc()
        code = 500
        response.status_code = 500

    # print(standardized_outputs)
    # 转换输出结果中的类型
    return RelationExtractionResponse(text=data.text,
                                      result=standardized_outputs,
                                      message=message,
                                      code=code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)

```

# 测试

```python
import json
import time

import requests

data = {
    "text": "",
    "labels": ["人名"]
}

url = "http://192.168.11.160:7777/extract_relation"
t1 = time.time()
response = requests.post(url=url, data=json.dumps(data))
t2 = time.time()
print(t2-t1)
print(response.text)
```

# 人名脱敏

```python
import pandas as pd
import json
import time

import requests

from tqdm import tqdm

def get_data():
    data = pd.read_excel("data/脱敏后数据.xlsx")
    res = [d[1] for d in data.iterrows()]
    return res


def get_ner(text):
    data = {
        "text": text,
        "labels": ["人名"]
    }
    # print(text)
    url = "http://192.168.11.160:7777/extract_relation"
    response = requests.post(url=url, data=json.dumps(data))
    response = response.json()
    result = response["result"]
    tmp = []
    if "人名" in result:
        result = result["人名"]
        result = sorted(result, key=lambda x:x["start"])
        # print(result)
        t_start= None
        t_end = None
        for d in result:
            start = d["start"]
            end = d["end"]
            tmp.append(text[t_start:start] + text[start:start+1] + "某")
            t_start = end
            t_end = end
        if t_end != len(text):
            # print(text[t_end:])
            tmp.append(text[t_end:])
        tmp = "".join(tmp)
    else:
        tmp = text
    return tmp


def process_one(args):
    i, d = args
    id = d["CASECODE"]
    content = d["RQSTCONTENT"]
    res = get_ner(content)
    t = {}
    t["CASECODE"] = id
    t["RQSTCONTENT"] = content
    t["人名脱敏"] = res
    return t


import concurrent.futures

num_workers = 128

data = get_data()

tt = []
with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
    # 提交任务并获得 future 对象
    futures = [executor.submit(process_one, (i, t)) for i, t in enumerate(data)]
    # for future in tqdm(concurrent.futures.as_completed(futures)):
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
        result = future.result()
        # print(result)
        # print("="*100)
        tt.append(result)

tt = pd.DataFrame(tt)
tt.to_excel("人名脱敏后数据.xlsx")
```

