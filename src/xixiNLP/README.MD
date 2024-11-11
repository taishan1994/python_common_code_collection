python项目的一个基本架构，主要是能够在xixinlp的同级目录下可以以```import xixinlp as xixi```的方式来使用我们定义的模块。

```python
# coding="utf-8"
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

"""
hello world
hello world
2022-12-27 17:21:16 INFO seg_jieba: ['北京市', '长江大桥']
[seg_jieba] run time is 0.0000
['北京市', '长江大桥']
1672132876.0586689
2022-12-27 17:21:16 INFO seg_jieba2: ['北京市', '长江大桥']
[seg_jieba2] run time is 0.0000
['北京市', '长江大桥']
"""
```

还可以参考：`https://github.com/roboflow/template-python`
