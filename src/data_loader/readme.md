# 方法一：通过python文件加载数据

```python
import datasets


myDataset = datasets.load_dataset("get_data.py", streaming=True)["train"]

print(myDataset)

```

```python
import glob
import datasets


myDataset = datasets.load_dataset("json", data_files=glob.glob("data/*.json"))
print(myDataset)

```

## 对数据集进行shuffle

```python
train_set = myDataset.shuffle(seed=123)
```