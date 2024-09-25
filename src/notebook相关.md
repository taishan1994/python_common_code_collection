# 启动

````shell
# pip install notebook
python -m notebook --ip=0.0.0.0 --port=xxx
````

## 展示图片

```python
import matplotlib.pyplot as plt
from PIL import Image

def show_imgs(imgs, titles=None):
    if len(imgs) == 1:
        plt.imshow(Image.open(imgs[0]))
        if titles:
            plt.title(titles[0])
        plt.show()
        return 0
	# fig, axs = plt.subplots(1, len(imgs), figsize=(120,60))
    fig, axs = plt.subplots(1, len(imgs), figsize=(60,30))
    for idx, img in enumerate(imgs):
        if isinstance(img, str):
            img = Image.open(img)
        axs[idx].imshow(img)
        axs[idx].axis('off')
        if titles:
            axs[idx].set_title(titles[idx])
    plt.show()
    return 0

```

## 图片去重

```python
# pip install imagededup
# 先将图片转换为Hash码
import json
import faiss

from tqdm import tqdm
from pprint import pprint
from pathlib import Path

# 存储图片及其对应的hash码的json
phash_file = "xx.json"

with open(phash_file, 'r') as fp:
    phash_dict = json.load(fp)

tmp = []
for k,v in phash_dict.items():
    img_name = k
    hash_bin = int(v, 16)
    hash_bin = bin(hash_bin)[2:]
    hash_bin = [float(i) for i in hash_bin]
    # print(hash_bin)
    tmp.append(hash_bin)

import numpy as np
tmp = np.array(tmp, dtype='float32')
print(tmp.shape)

idx2img = {}
for i,k in enumerate(list(phash_dict.keys())):
    idx2img[i]=k

import faiss
d = 64
index = faiss.IndexFlatL2(d)
# index = faiss.index_cpu_to_all_gpus(index)
index.add(tmp)

from tqdm import tqdm

threshold = 8.1
should_remove = set()
is_retriever = set()


batch_size = 1  # 例如一次传输 10000 条数据
for ind in tqdm(range(10000, tmp.shape[0])):
    t_img_path = idx2img[ind]
    if t_img_path in is_retriever:
        continue
    # range_search和search的区别
    _, D, I = index.range_search(np.array([tmp[ind]]), threshold)
    D = D.tolist()
    I = I.tolist()
    t = []
    if len(D) == 0:
        continue
    print(D, I)
    for i,d in zip(I, D):
        img_path = idx2img[i]
        if img_path not in is_retriever:
            is_retriever.add(img_path)
        if img_path not in should_remove:
            should_remove.add(img_path)
            
        img_path_real = img2path[img_path] + "/" + img_path
        print(img_path_real)
        t.append(img_path_real)
    if len(t) > 1:
        show_imgs(t)
        break

print(len(is_retriever))
print(len(should_remove))
```

