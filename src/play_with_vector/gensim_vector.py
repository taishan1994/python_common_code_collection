from pprint import pprint
from gensim.models import KeyedVectors

from utils import print_run_time


@print_run_time
def load_model():
    path = "data/100000-small.txt"
    """
    txt里面的一般格式：
        第一行：向量总数 向量维度
        下面的每一行：词 词向量
        每一个元素之间用空格隔开
    """
    wv_from_text = KeyedVectors.load_word2vec_format(path, binary=False)
    return wv_from_text


# 计算词语之间的相似度
@print_run_time
def cal_sim(wv_from_text, word):
    wv_from_text.init_sims(replace=True)
    if word in wv_from_text.wv.vocab.keys():
        vec = wv_from_text[word]
        pprint(wv_from_text.most_similar(positive=[vec], topn=20))
    else:
        print("没找到")


# 计算词之间的距离
@print_run_time
def cal_distance(wv_from_text, word1, word2):
    """这里指的距离，并不是近义词或者反义词，只是句中该处是否可被另一个词替换的可能性"""
    print("word1：{} word2：{} distance：{}".format(
        word1, word2, wv_from_text.distance(word1, word2)
    ))


# 计算两个词语之间的相似度
@print_run_time
def cal_similary(wv_from_text, word1, word2):
    """这里指的距离，并不是近义词或者反义词，只是句中该处是否可被另一个词替换的可能性"""
    print("word1：{} word2：{} similary：{}".format(
        word1, word2, wv_from_text.similarity(word1, word2)
    ))


# 计算两个字符串列表之间的相似度
@print_run_time
def cal_n_similarity(wv_from_text, word_list1, word_list2):
    """这里指的距离，并不是近义词或者反义词，只是句中该处是否可被另一个词替换的可能性"""
    print("word_list1：{} word_list2：{} similary：{}".format(
        word_list1, word_list2, wv_from_text.n_similarity(word_list1, word_list2)
    ))


def main():
    wv_from_text = load_model()
    cal_sim(wv_from_text, "王者")

    cal_distance(wv_from_text, "喜欢", "爱")
    cal_distance(wv_from_text, "喜欢", "讨厌")
    cal_distance(wv_from_text, "喜欢", "西瓜")
    cal_distance(wv_from_text, "喜欢", "稀饭")

    cal_similary(wv_from_text, "喜欢", "爱")
    cal_similary(wv_from_text, "喜欢", "讨厌")

    cal_n_similarity(wv_from_text, ["风景", "怡人"], ["景色", "优美"])
    cal_n_similarity(wv_from_text, ["风景", "怡人"], ["上", "厕所"])

    """
    [load_model] run time is 13.9975s
    [('王者', 0.9999999403953552),
     ('霸主', 0.7173691391944885),
     ('之王', 0.6815822720527649),
     ('大魔王', 0.6740668416023254),
     ('一哥', 0.6711729764938354),
     ('战神', 0.6695720553398132),
     ('无敌', 0.6684187054634094),
     ('排位', 0.6682911515235901),
     ('称霸', 0.6467258930206299),
     ('至尊', 0.6451884508132935),
     ('王者荣耀', 0.641219973564148),
     ('段位', 0.6411482691764832),
     ('王者归来', 0.6365198493003845),
     ('魔王', 0.6318097114562988),
     ('超神', 0.6295116543769836),
     ('lol', 0.62571120262146),
     ('王座', 0.6213178634643555),
     ('排位赛', 0.6197971105575562),
     ('召唤师', 0.6183013319969177),
     ('荣耀', 0.6182342171669006)]
    [cal_sim] run time is 0.0462s
    word1：喜欢 word2：爱 distance：0.29567837715148926
    [cal_distance] run time is 0.0000s
    word1：喜欢 word2：讨厌 distance：0.2994500398635864
    [cal_distance] run time is 0.0000s
    word1：喜欢 word2：西瓜 distance：0.6705849468708038
    [cal_distance] run time is 0.0000s
    word1：喜欢 word2：稀饭 distance：0.5587387084960938
    [cal_distance] run time is 0.0000s
    word1：喜欢 word2：爱 similary：0.7043216228485107
    [cal_similary] run time is 0.0000s
    word1：喜欢 word2：讨厌 similary：0.7005499601364136
    [cal_similary] run time is 0.0000s
    word1：['风景', '怡人'] word2：['景色', '优美'] similary：0.8653326034545898
    [cal_n_similarity] run time is 0.0000s
    word1：['风景', '怡人'] word2：['上', '厕所'] similary：0.43212899565696716
    [cal_n_similarity] run time is 0.0000s
    """


if __name__ == '__main__':
    main()
