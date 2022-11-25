# coding=utf8
"""
代码参考：https://github.com/china-ai-law-challenge/CAIL2022/blob/main/lajs/baseline/main.py
"""
import json
import os
import pickle

from gensim.summarization import bm25
import jieba
import numpy as np
from tqdm import tqdm


def get_stopwords():
    with open("data/stopword.txt", "r", encoding="utf-8") as fp:
        words = fp.read().split("\n")
    words.extend(['.', '（', '）', '-'])
    return words


def get_shence_contents():
    path = "data/shence_data/shence_data.json"
    with open(path, "r", encoding="utf-8") as fp:
        lines = json.loads(fp.read())
    return lines


def save_model(model, path):
    with open(path, "wb") as fp:
        pickle.dump(model, fp)


def load_model(path):
    with open(path, "rb") as fp:
        return_model = pickle.load(fp)
    return return_model

def get_docid2content():
    lines = get_shence_contents()
    docid2content = {i:line["title"]+line["content"] for i,line in enumerate(lines)}
    return docid2content

def get_bm25_model():
    print('begin...')
    stopwords = get_stopwords()
    lines = get_shence_contents()
    result = {}
    corpus = []
    for i, line in tqdm(enumerate(lines), total=len(lines)):
        title = line["title"]
        content = line["content"]
        result[i] = []
        words = jieba.lcut(title + content, cut_all=False)
        corpus.append([i for i in words if not i in stopwords])
    bm25Model = bm25.BM25(corpus)
    save_model(bm25Model, "bm25_shence.model")


def get_sim(query, bm25Model, docid2content, topk=10):
    ori_query = query
    query = jieba.lcut(query, cut_all=False)
    query = [i for i in query if not i in stopwords]
    raw_rank_index = np.array(bm25Model.get_scores(query)).argsort().tolist()[::-1][:topk]
    results = [docid2content[i] for i in raw_rank_index]
    print("query：", ori_query)
    print("="*100)
    for res in results:
        print(res)


if __name__ == "__main__":
    stopwords = get_stopwords()
    docid2content = get_docid2content()
    bm25Model = load_model("bm25_shence.model")
    query = "我喜欢吃水饺"

    get_sim(
        query,
        bm25Model,
        docid2content,
        topk=10
    )

