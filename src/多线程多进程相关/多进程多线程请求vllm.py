import concurrent.futures
import json
import multiprocessing
import os
import glob
import random
import traceback

from openai import OpenAI

import mimetypes
import base64

from tqdm import tqdm


def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"


def judge(args):
    i, (d, out_file, client) = args
    img_path = os.path.join(root_path)
    img_name = d["img_path"]
    real_path = os.path.join(img_path, img_name)
    try:
        img_bs64 = encode_image(real_path)

        chat_response = client.chat.completions.create(
            model="Qwen2-VL-72B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            # "text": "请尽可能简短描述下该图片，主要关注和人相关的属性或者物品。"
                            "text": "请尽可能简短描述下图片中人的属性，性别可以是男女，年龄可以是中年人、老年人、小孩等，例如：穿着白色短袖衬衫和棕色长裤，白色运动鞋，戴着手表的男人。\n两个穿黑色衣服黑色裤子的人。\n一个黄色长头发的女人等。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_bs64,
                            },

                        }
                    ]
                },
            ],
            temperature=0.3
        )
        # print(chat_response)
        tmp = {}
        tmp["image_name"] = img_name
        tmp["image_path"] = root_path
        tmp["qwen2-vl-72B"] = chat_response.choices[0].message.content
    except Exception as e:
        print(e)
        tmp = {}
        tmp["image_name"] = img_name
        tmp["image_path"] = root_path
        tmp["qwen2-vl-72B"] = "出错"
    if i == 0:
        print(tmp)
    return tmp


def process_data(batch_data, out_file, client, rank):
    num_workers = 16
    save_path = os.path.join(save_dir, f"{rank}.json")
    save_file = open(save_path, "a", buffering=1)
    with concurrent.futures.ThreadPoolExecutor(num_workers) as excutor:
        futures = [excutor.submit(judge, (i, (d, out_file, client))) for i, d in enumerate(batch_data)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"rank-{rank + 1}"):
            result = future.result()
            caption = result["qwen2-vl-72B"]
            if caption != "出错" and len(caption) < 256:
                save_file.write(json.dumps(result, ensure_ascii=False) + "\n")


def main(input_data, urls):
    num_processes = len(urls)
    sample_per_batch = (len(input_data) + len(urls) - 1) // len(urls)

    processes = []
    for i in range(num_processes):
        batch_data = data[i * sample_per_batch:(i + 1) * sample_per_batch]

        openai_api_key = "EMPTY"
        # openai_api_base = "http://192.168.112.6:8888/v1"
        # openai_api_base = "http://192.168.112.75:4788/v1"
        openai_api_base = urls[i]

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        process = multiprocessing.Process(target=process_data, args=(batch_data, "", client, i),
                                          name=f"Process-{i + 1}")
        processes.append(process)
        process.start()

    for pro in processes:
        pro.join()


if __name__ == '__main__':
    # 读取数据
    dep_files = "/huawei-data/FM/wangtian/datasets/image_dedup/dedup_output/gob_subimg_full_path_1011_dedup_result.json"
    with open(dep_files, "r") as fp:
        data = json.loads(fp.read())

    print("总共有数据：", len(data))

    save_dir = "/huawei-data/FM/gob/语义搜/test_multimodal_model/caption_data/shu_cang"
    image_name_set = set()

    files = glob.glob(save_dir + "/*.json")
    for file in files:
        with open(file, "r") as fp:
            has_caption_data = fp.read().strip().split("\n")
        for d in has_caption_data:
            d = json.loads(d)
            image_name = d["image_name"]
            image_name_set.add(image_name)

    print("已有数据：", len(image_name_set))

    input_data = []
    for d in tqdm(data, total=len(data)):
        img_name = d["img_path"]
        if img_name in image_name_set:
            continue
        input_data.append(d)

    print("还剩余数据：", len(input_data))

    root_path = "/huawei-data/FM/gob/yys/yys_0914/chengdu_laoshan/person/"
    # 没每一部分数据分给不同的url

    # 每一个url的数据进行多线程处理
    urls = ["http://192.168.112.172:8888/v1",
            "http://192.168.112.174:8888/v1",
            "http://192.168.112.161:8888/v1",
            "http://192.168.112.153:8888/v1",
            "http://192.168.112.169:8888/v1",
            "http://192.168.112.167:8888/v1",
            "http://192.168.112.144:8888/v1",
            "http://192.168.112.150:8888/v1",
            "http://192.168.112.190:8888/v1",
            "http://192.168.112.193:8888/v1",
            ]
    # 断点续传
    main(input_data, urls)
