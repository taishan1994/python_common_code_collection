"""
python elasticsearch==7.6.0
elasticsearch=7.6.0
"""
import time
from pprint import pprint

import pandas as pd
from elasticsearch import Elasticsearch, helpers


def get_data():
    data = pd.read_csv("data.csv", header=0, encoding="gb2312")
    return data


def timer(func):
    """用于计算一个函数的耗时"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        print("共消耗%.3f秒" % (time.time() - start_time))
        return res

    return wrapper


class Args:
    addr = "http://192.168.188.1:9200"


class ESProcessor:
    def __init__(self, args):
        self.es = Elasticsearch(args.addr)
        print(self.es.info())

    def create_index(self, index, body=None):
        self.es.indices.create(index=index, body=body, ignore=400)

    def delete_index(self, index):
        self.es.indices.delete(index=index)

    def insert_one(self, index, body):
        self.es.index(index=index, body=body)

    def insert_many(self, index, insert_data, batch_size=10000):
        batch_data = []
        for d in insert_data.iterrows():
            d = d[1]
            ajbh = d["AJBH"]
            ajbh = ajbh if str(ajbh).lower() != "nan" else ""
            jyaq = d["JYAQ"]
            jyaq = jyaq if str(jyaq).lower() != "nan" else ""
            fadd = d["FADD"]
            fadd = fadd if str(fadd).lower() != "nan" else ""
            batch_data.append({
                "_index": index,
                "_source": {"ajbh": ajbh, "jyaq": jyaq, "fadd": fadd}
            })
            if len(batch_data) == batch_size:
                helpers.bulk(self.es, batch_data)
                batch_data = []
        if len(batch_data) != 0:
            helpers.bulk(self.es, batch_data)

    def search(self, index, search_body, *args, **kwargs):
        return self.es.search(index=index, body=search_body, *args, **kwargs)

    def get_by_id(self, index, id):
        return self.es.get(index=index, id=id)

    def update_by_id(self, index, id, update_body):
        return self.es.update(index=index, id=id, body=update_body)

    def delete_by_id(self, index, id):
        return self.es.delete(index=index, id=id)

args = Args()
esProcessor = ESProcessor(args)

"""
data = get_data()
print("总共有数据：{}".format(len(data)))
create_body = {
    "settings": {},
    "mappings": {
        "properties": {
            "ajbh": {
                "type": "keyword",
                "index": "true",
            },
            "jyaq": {
                "type": "text",
                "index": "true",
            },
            "fadd": {
                "type": "text",
                "index": "true",
            }
        }
    }
}
esProcessor.delete_index(index="t_ajxx")
esProcessor.create_index(index="t_ajxx", body=create_body)
esProcessor.insert_many(index="t_ajxx", insert_data=data)
"""

# 全词搜索并高亮
query_body = {
    "query": {
        "match_phrase": {
            "fadd": "龙华区",
        },
    },
    "highlight": {
        "boundary_scanner_locale": "zh_CN",
        "boundary_scanner": "word",
        "fragmenter": "span",
        "fields": {
            "fadd": {
                "pre_tags": [
                    "<em>"
                ],
                "post_tags": [
                    "</em>"
                ]
            }
        }
    }
}

res = esProcessor.search(index="t_ajxx", search_body=query_body)
pprint(res)

search_body = {
    "query": {
        "match_phrase": {
            "ajbh": "A4403116000002022016003"
        }
    }
}
res = esProcessor.search(index="t_ajxx", search_body=search_body)
pprint(res)

res = esProcessor.get_by_id(index="t_ajxx", id="ilfYSYYBL6FT4ZX6O0VS")
pprint(res)
