import jieba

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


def sim(a, b, cut_word=True):
    if cut_word:
        a = jieba.lcut(a, cut_all=False)
        b = jieba.lcut(b, cut_all=False)
    else:
        a = [i for i in a]
        b = [i for i in b]
    c = list(set(a + b))
    a1 = np.array([1 if i in a else 0 for i in c])
    b1 = np.array([1 if i in b else 0 for i in c])
    num = float(np.dot(a1, b1))  # 向量点乘
    denom = np.linalg.norm(a1) * np.linalg.norm(b1)  # 求模长的乘积
    return (num / denom) if denom != 0 else 0


total = len(documents)
for i in range(total):
    for j in range(total):
        if i != j:
            s = sim(documents[i], documents[j], cut_word=False)
            if s > 0.5:
                print(documents[i], documents[j])
                print(s)
    print('=' * 100)
