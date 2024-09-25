"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
It also demonstrates how to stream data which is helpful in case you don't
want to wait for an extremely large dataset to download, or if you want to
limit the amount of memory used. More info about dataset streaming:
https://huggingface.co/docs/datasets/stream
"""
import json
import logging
import numpy as np

from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_transformers import LoggingHandler, SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

# Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == "__main__":
    # Set params
    data_stream_size = 16384  # Size of the data that is loaded into memory at once
    chunk_size = 1024  # Size of the chunks that are sent to each process
    encode_batch_size = 128  # Batch size of the model

    model_name = ""
    model_dict = {}

    with open("data/contents.json", "r", encoding="utf-8") as fp:
        data = json.loads(fp.read())

    # data = data[:1]
    # print(data)
    data = {"content": data}
    dataset = Dataset.from_dict(data, split="train")

    print(dataset)
    # Load a large dataset in streaming mode. more info: https://huggingface.co/docs/datasets/stream
    # dataset = load_dataset("yahoo_answers_topics", split="train", streaming=True)
    dataloader = DataLoader(dataset.with_format("torch"), batch_size=data_stream_size)

    # Define the model
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model_dir = "./model_hub/embedding/AI-ModelScope/bge-large-zh-v1___5"
    model_dir = "./model_hub/embedding/AI-ModelScope/gte-large-zh"
    model_dir = "./model_hub/embedding/maidalun/bce-embedding-base_v1"
    # from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_dir)

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    res = []

    for i, batch in enumerate(tqdm(dataloader)):
        # Compute the embeddings using the multi-process pool
        sentences = batch["content"]
        # batch_emb = model.encode_multi_process(sentences, pool, chunk_size=chunk_size, batch_size=encode_batch_size, normalize_embeddings=True, prompt="为这个句子生成表示以用于检索相关文章：")
        batch_emb = model.encode_multi_process(sentences, pool, chunk_size=chunk_size, batch_size=encode_batch_size,
                                               normalize_embeddings=True)
        print("Embeddings computed for 1 batch. Shape:", batch_emb.shape)
        # print(batch_emb)
        if len(res) == 0:
            res = batch_emb
        else:
            res = np.concatenate((res, batch_emb), axis=0)

    # Optional: Stop the processes in the pool
    model.stop_multi_process_pool(pool)
    print(res.shape)
    np.save("embeddings/bce.npy", res)
