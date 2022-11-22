# 说明

快速计算词向量之间的相似度。

# 数据

可以去这两个地方下载：

https://ai.tencent.com/ailab/nlp/en/download.html <br>
https://github.com/cliuxinxin/TX-WORD2VEC-SMALL

这里使用的是腾讯词向量的100000-small.txt

# 计算时间

| 模型             | 时间    |
| ---------------- | ------- |
| gensim           | 0.0462s |
| sklearn_kdtree   | 0.0318s |
| sklearn_balltree | 0.0444s |
| faiss            | 0.0169s |
| annoy            | 0.0000s |

# 参考

部分代码参考：[topk相似度性能比较](https://blog.csdn.net/luoyexuge/article/details/84235421)
