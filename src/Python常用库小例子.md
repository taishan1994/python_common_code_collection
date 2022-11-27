#### 检测文本的编码
```python
# pip install chardet
import chardet
data = '离离原上草，一岁一枯荣'.encode('gb2312')
chardet.detect(data)
```
#### 终端格式化输出
```python
# pip install prettytable
from prettytable import PrettyTable
## 按行添加数据
tb = pt.PrettyTable()
tb.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
tb.add_row(["Adelaide",1295, 1158259, 600.5])
tb.add_row(["Brisbane",5905, 1857594, 1146.4])
tb.add_row(["Darwin", 112, 120900, 1714.7])
tb.add_row(["Hobart", 1357, 205556,619.5])

"""
+-----------+------+------------+-----------------+
| City name | Area | Population | Annual Rainfall |
+-----------+------+------------+-----------------+
|  Adelaide | 1295 |  1158259   |      600.5      |
|  Brisbane | 5905 |  1857594   |      1146.4     |
|   Darwin  | 112  |   120900   |      1714.7     |
|   Hobart  | 1357 |   205556   |      619.5      |
+-----------+------+------------+-----------------+
"""
```
#### 比较两个文本之间的差异
```python
import difflib
text1 = '''  1. Beautiful is better than ugly.
       2. Explicit is better than implicit.
       3. Simple is better than complex.
       4. Complex is better than complicated.
		'''.splitlines(keepends=True)
text2 = '''  1. Beautiful is better than ugly.
       3.   Simple is better than complex.
       4. Complicated is better than complex.
       5. Flat is better than nested.
     '''.splitlines(keepends=True)

d = difflib.Differ()
print(''.join(list(d.compare(text1,text2))))

d = difflib.HtmlDiff()
htmlContent = d.make_file(text1,text2)
# print(htmlContent)
with open('diff.html','w') as f:
    f.write(htmlContent)
```
#### 计算编辑距离
```python
pip install python-Levenshtein
import Levenshtein.ratio
Levenshtein.ratio('hello', 'world')  # (10 -8) / 10
```
#### 字符串模糊匹配
 ```python
 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple FuzzyWuzzy
 ```
 #### AC自动机算法
 ```python
 pip install esmre
 import esm
index = esm.Index()
index.enter('保罗')
index.enter('小卡')
index.enter('贝弗利')
index.fix()
index.query("""NBA季后赛西部决赛，快船与太阳移师洛杉矶展开了他们系列赛第三场较量，上一场太阳凭借艾顿的空接绝杀惊险胜出，此役保罗火线复出，而小卡则继续缺阵。首节开局两队势均力敌，但保罗和布克单节一分未得的拉胯表现让太阳陷入困境，快船趁机在节末打出一波9-2稍稍拉开比分，次节快船替补球员得分乏术，太阳抓住机会打出14-4的攻击波反超比分，布克和保罗先后找回手感，纵使乔治重新登场后状态火热，太阳也依旧带着2分的优势结束上半场。下半场太阳的进攻突然断电，快船则在曼恩和乔治的引领下打出一波21-3的攻击狂潮彻底掌控场上局势，末节快船在领先到18分后略有放松，太阳一波12-0看到了翻盘的希望，关键时刻雷吉和贝弗利接管比赛，正是他们出色的发挥为球队锁定胜局，最终快船主场106-92击败太阳，将总比分扳成1-2。""")
 ```
 #### 汉字转拼音
 ```python
 pip install xpinyin
 ```


#### cachetools
```python
from cachetools import cached, LRUCache
import time

# cache using LRUCache
@cached(cache = LRUCache(maxsize = 3))
def myfun(n):

    # This delay resembles some task
    s = time.time()
    time.sleep(n)
    print("\nTime Taken: ", time.time() - s)
    return (f"I am executed: {n}")

# Takes 3 seconds
print(myfun(3))

# Takes no time
print(myfun(3))

# Takes 2 seconds
print(myfun(2))

# Takes 1 second
print(myfun(1))

# Takes 4 seconds
print(myfun(4))

# Takes no time
print(myfun(1))

# Takes 3 seconds because maxsize = 3 
# and the 3 recent used functions had 1,
# 2 and 4.
print(myfun(3))
```
