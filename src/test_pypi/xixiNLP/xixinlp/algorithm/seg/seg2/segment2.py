import jieba

jieba.initialize()

from xixinlp import logging
from xixinlp.utils.time_it import print_run_time

print("hello world")

@print_run_time
def seg_jieba2(text):
    res = jieba.lcut(text, cut_all=False)
    logging.info(res)
    return res


