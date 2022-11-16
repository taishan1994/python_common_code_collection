import requests
import time

start = time.time()
url = "http://192.168.137.112:9999/"
res = requests.get(url)
print(res.text)
end = time.time()
print("耗时:{}s".format(end - start))
