#### 检测文本的编码
```python
# pip install chardet
import chardet
data = '离离原上草，一岁一枯荣'.encode('gb2312')
chardet.detect(data)
```
