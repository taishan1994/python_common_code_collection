import numpy as np
from sklearn.neighbors import KDTree, BallTree
from pprint import pprint
from utils import load_vectors, print_run_time


@print_run_time
def search(model,
           word,
           data,
           word2id,
           id2word,
           k=10,
           ):
    vector = data[word2id[word]]
    dists, inds = model.query([vector], k)
    return list(zip([id2word[idx] for idx in inds[0][1:]], dists[0][1:]))


if __name__ == '__main__':
    path = "data/100000-small.txt"
    return_data = load_vectors(path)

    data = return_data["data"]
    word2id = return_data["word2id"]
    id2word = return_data["id2word"]

    data = np.array(data)
    balltree = BallTree(data, leaf_size=100)
    kdtree = KDTree(data, leaf_size=100)

    if type == "kd":
        model = kdtree
    else:
        model = balltree

    pprint(search(model, "王者", data, word2id, id2word, k=10, type="kd"))
    pprint(search(model, "王者", data, word2id, id2word, k=10, type="ball"))

    """
    [search] run time is 0.0318s
    [('霸主', 0.7518388563914274),
     ('之王', 0.7980198103207503),
     ('大魔王', 0.8073822793697959),
     ('一哥', 0.8109587098492215),
     ('战神', 0.8129303229285683),
     ('无敌', 0.8143480098416078),
     ('排位', 0.8145046379406339),
     ('称霸', 0.8405643447926074),
     ('至尊', 0.8423915197575921)]
    [search] run time is 0.0444s
    [('霸主', 0.7518388563914274),
     ('之王', 0.7980198103207503),
     ('大魔王', 0.8073822793697959),
     ('一哥', 0.8109587098492215),
     ('战神', 0.8129303229285683),
     ('无敌', 0.8143480098416078),
     ('排位', 0.8145046379406339),
     ('称霸', 0.8405643447926074),
     ('至尊', 0.8423915197575921)]
    """
