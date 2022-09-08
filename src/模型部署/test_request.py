# coding=utf-8
import requests
import time

params = {
   "text": "武汉市长江大桥来我公司进行考察。"
}

url = "http://你的ip地址:8080/stream"
start = time.time()
res = requests.post(url=url, json=params)
print(res.text)
end = time.time()
print("耗时：{}s".format(end - start))
