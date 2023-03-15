import time
from gensim.models import KeyedVectors


def print_run_time(func):
    """时间装饰器"""

    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print('[%s] run time is %.4fs' % (func.__name__, time.time() - local_time))
        return res

    return wrapper


@print_run_time
def load_txt_data(path):
    with open(path, "r", encoding="utf-8") as fp:
        data = fp.read().strip().split("\n")
    des = data[0].split(" ")
    vectors = data[1:]
    print("总共有：{}个词向量，维度是：{}".format(des[0], des[1]))
    return vectors


def load_vectors(path):
    wv_from_text = KeyedVectors.load_word2vec_format(path, binary=False)
    wv_from_text.init_sims(replace=True)
    word2idx = {}
    idx2word = {}
    data = []
    for ind, key in enumerate(wv_from_text.vocab.keys()):
        data.append(wv_from_text[key])
        word2idx[key] = ind
        idx2word[ind] = key

    return_data = {
        "word2id": word2idx,
        "id2word": idx2word,
        "data": data,
    }
    return return_data


if __name__ == '__main__':
    path = "data/100000-small.txt"
    vectors = load_txt_data(path)
    print(vectors[0].split(" "))

    # load_vectors(path)
