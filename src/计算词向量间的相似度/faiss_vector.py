import faiss
import numpy as np
from pprint import pprint
from utils import load_vectors, print_run_time

"""pip install faiss-cpu"""


@print_run_time
def search(
        model,
        word,
        data,
        word2id,
        id2word,
        k=10):
    vector = data[word2id[word]]
    dists, inds = model.search(vector.reshape(-1, 200), k)
    return list(zip([id2word[idx] for idx in inds[0][1:]], dists[0][1:]))


if __name__ == '__main__':
    path = "data/100000-small.txt"
    return_data = load_vectors(path)

    data = return_data["data"]
    word2id = return_data["word2id"]
    id2word = return_data["id2word"]

    data = np.array(data)

    quantizer = faiss.IndexFlatIP(200)  # the other index，需要以其他index作为基础
    faiss_index = faiss.IndexIVFFlat(quantizer, 200, 120, faiss.METRIC_L2)
    faiss_index.train(data)
    faiss_index.nprobe = 80
    faiss_index.add(data)  # add may be a bit slower as well

    pprint(search(faiss_index, "王者", data, word2id, id2word, k=10))

    """
    [search] run time is 0.0169s
    [('霸主', 0.5652616),
     ('之王', 0.6368352),
     ('大魔王', 0.6518659),
     ('一哥', 0.6576541),
     ('战神', 0.6608557),
     ('无敌', 0.6631627),
     ('排位', 0.66341764),
     ('称霸', 0.7065484),
     ('至尊', 0.7096234)]
    """
