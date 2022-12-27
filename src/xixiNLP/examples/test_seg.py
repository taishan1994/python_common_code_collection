# coding="utf-8"
import sys
sys.path.append("..")
import xixinlp as xixi

text = "北京市长江大桥"
seg_text = xixi.seg.seg_jieba(text)
print(seg_text)

import time
# cur_time = xixi.timeParser.cur_time
# print(cur_time)
cur_time = xixi.timeParser(time.time())
print(cur_time)

text = "北京市长江大桥"
seg_text = xixi.seg.seg2.seg_jieba2(text)
print(seg_text)
