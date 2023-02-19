测试上传包到pypi模板。

执行步骤：

首先可以https://test.pypi.org/ （用于测试）和https://pypi.org/ （正式环境）上分别注册一个账号（可以相同），然后分别ADD API TOKEN。

1、按照项目修改setup.py里面的内容。

2、打包

```python
cd xixiNLP
python3 setup.py sdist bdist_wheel
```

3、windwos下在C:\Users\Administrator，lindex系统在~/下新建一个.pypirc，里面驶入：
```python
[distutils]
index-servers=testpypi

[testpypi]
username=__token__
password=自己添加的token
```

4、安装twine，并上传打好的包到testpypi

```python
python3 -m pip install --user --upgrade twine

python3 -m twine upload --repository testpypi dist/*
```

之后会提示在[xixinlp · TestPyPI](https://test.pypi.org/project/xixinlp/0.0.1/#files) 上查看。

5、安装包并测试

```python
pip install -i https://test.pypi.org/simple/ xixinlp==0.0.1

import xixinlp as xixi
xixi.seg.seg_jieba("武汉市长江大桥")
```

5、上传到pypi

```python
python3 -m twine upload dist/*
```

同理也可以测试。

