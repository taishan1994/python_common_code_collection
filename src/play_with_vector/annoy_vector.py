from pprint import pprint
from annoy import AnnoyIndex

from utils import load_vectors, print_run_time


@print_run_time
def search(model,
           word,
           word2id,
           id2word,
           k=10):
    index = word2id[word]
    result = model.get_nns_by_item(index, k)
    word_result = [id2word[idx] for idx in result[1:]]
    return word_result


if __name__ == '__main__':
    path = "data/100000-small.txt"
    return_data = load_vectors(path)

    data = return_data["data"]
    word2id = return_data["word2id"]
    id2word = return_data["id2word"]
    annoy_model = AnnoyIndex(200, 'angular')
    for i, vector in enumerate(data):
        annoy_model.add_item(i, vector)
    annoy_model.build(100)
    pprint(search(annoy_model, "王者", word2id, id2word, k=10))

    """
    [search] run time is 0.0000s
    ['霸主', '之王', '大魔王', '战神', '无敌', '称霸', '段位', '王者归来', '超神']
    """
