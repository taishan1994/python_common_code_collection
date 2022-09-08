import jieba
import json
import numpy as np
from gensim import corpora, models, similarities

documents = [
    "我不喜欢上海",
    "上海是一个好地方",
    "北京是一个好地方",
    "上海好吃的在哪里",
    "上海好玩的在哪里",
    "上海是好地方",
    "上海路和上海人",
    "喜欢小吃"
]

texts = [jieba.lcut(text, cut_all=False) for text in documents]  # 分词
dictionary = corpora.Dictionary(texts)  # 构建词表
print(dictionary)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]  # 

tfidf = models.TfidfModel(corpus)

corpus_tfidf  = tfidf[corpus]
index = similarities.MatrixSimilarity(corpus_tfidf)

for j,c in enumerate(corpus):
    print("query:", documents[j])
    vec = tfidf[c]
    score = index[vec]
    sim = np.argsort(-score)
    res = np.argsort(sim)
    tmp = [(s, ind) for ind, s in enumerate(res)]
    tmp = sorted(tmp, key=lambda x:x[0])
    tmp = [i[1] for i in tmp]
    for i in tmp:
        print(score[i], documents[i])
    print("*"*50)
   
# 可以进一步使用主题模型
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)
corpus_lsi = lsi[corpus_tfidf]
index = similarities.MatrixSimilarity(lsi[corpus])

query = "上海好地方"
query_bow = dictionary.doc2bow(jieba.lcut(query, cut_all=False))
query_lsi = lsi[query_bow]
sims = index[query_lsi]
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sort_sims)
