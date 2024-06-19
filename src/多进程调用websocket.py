import json
import random
import pandas as pd
import traceback

import websockets
from collections import Counter

from tqdm import tqdm

labels = ['劳动人事', '刑事', '邻里关系', '婚姻关系', '社会保障', '房产纠纷', '合同纠纷', '民事', '其他', '行政', '治安消防', '物业管理', '邻里纠纷', '交通事故',
          '消费维权', '婚姻家庭纠纷', '征收补偿', '子女抚养', '知识产权', '工伤事故', '债权债务', '侵权纠纷', '公司相关', '刑事犯罪', '金融保险', '房产宅基地纠纷', '遗产继承',
          '土地纠纷', '劳动争议纠纷', '建筑工程', '拖欠农民工工资纠纷', '国内仲裁', '其他纠纷', '国内公证', '老人赡养', '物业纠纷', '损害赔偿纠纷', '事故相关人员个体识别鉴定',
          '其他消费纠纷', '医患纠纷', '道路交通事故纠纷', '生产经营纠纷', '法医病理司法鉴定', '山林土地纠纷', '电子商务纠纷', '法医临床司法鉴定', '证券期货', '知识产权专利技术司法鉴定',
          '文书司法鉴定', '环境损害司法鉴定', '医疗纠纷', '旅游纠纷', '法医物证司法鉴定', '交通事故成因、车速、痕迹鉴定', '征地拆迁纠纷', '法医精神病司法鉴定', '环境污染纠纷',
          '电器产品性能鉴定', '涉台公证', '国际仲裁', '声像资料司法鉴定', '法医毒物司法鉴定', '机动车性能检测', '精神障碍医学鉴定', '涉外公证', '微量物证司法鉴定', '痕迹司法鉴定',
          '涉澳公证', '涉港公证', '机电产品性能鉴定']
labels_dict = {k: [] for k in labels}


def get_data():
    with open("360127.json", "r") as fp:
        data = json.loads(fp.read())
    random.shuffle(data)
    print(len(data))
    tmp = []
    types = []
    for i, d in enumerate(data):
        # print(d)
        dtype = d["type"]
        question = d["question"]
        detail = d["detail"]
        reply = d["reply"]
        t = d
        t["id"] = i
        if len(labels_dict[dtype]) <= 100:
            labels_dict[dtype].append(t)
            types.append(dtype)

    types = Counter(types)
    print(types)
    # print(res.keys())
    # print(len(res))
    for k, v in labels_dict.items():
        tmp += v

    return tmp


import asyncio
import websockets


async def process_element(args):
    i, d = args
    t = d
    id = d["id"]
    dtype = d["type"]
    question = d["question"]
    detail = d["detail"]
    reply = d["reply"]
    question = question + " " + detail
    message = "{\"opType\":null,\"fr\":\"brand\",\"message\":\"" + question + "\",\"sessionId\":\"\"}"
    print(message)
    answer = []
    for attempt in range(3):  # 重试3次
        try:
            async with websockets.connect("ws://ailegal.baidu.com/api/pc/dialog/aiAgent") as websocket:
                await websocket.send(message)
                while True:
                    response = await websocket.recv()
                    try:
                        data = json.loads(response)["data"]["data"]
                        content = data["content"]
                        answer.append(content)
                    except Exception as e:
                        print(traceback.format_exc())
                        break
                answer = "".join(answer)
                break  # 成功处理后退出重试循环
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed error: {e}")
            if attempt < 2:
                print("Retrying...")
                await asyncio.sleep(1)
            else:
                print("Failed after 3 attempts")
                answer = ["Connection failed"]

    if i == 0:
        print(t)
    t["法行宝回答"] = answer
    return t


def run_asyncio_task(element):
    return asyncio.run(process_element(element))


if __name__ == '__main__':
    import concurrent.futures

    data = get_data()
    num_workers = 10
    res = []
    with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
        # 提交任务并获得 future 对象
        futures = [executor.submit(run_asyncio_task, (i, t)) for i, t in enumerate(data)]
        # for future in tqdm(concurrent.futures.as_completed(futures)):
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
            result = future.result()
            print(result)
            res.append(result)
    res = pd.DataFrame(res)
    res.to_excel("法行宝回答.xlsx")
