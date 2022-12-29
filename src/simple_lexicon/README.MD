![img](https://pic3.zhimg.com/80/v2-64e25e1a638e2328591d967c075a7a7e_1440w.webp)

对每一个字进行BMES词汇增强：

- B：该字位于词的开头
- M：该字位于词的中间
- E：该字位于词的结尾
- S：该字就是词

结果：

```python
[CLS] ['null', 'null', 'null', 'null']
武 ['武汉市', 'null', 'null', 'null']
汉 ['null', '武汉市', '武汉', 'null']
市 ['市长', 'null', '武汉市', 'null']
长 ['长江大桥', 'null', '市长', 'null']
江 ['null', '长江大桥', '长江', 'null']
大 ['大桥', '长江大桥', 'null', 'null']
桥 ['null', 'null', '大桥', 'null']
[SEP] ['null', 'null', 'null', 'null']
```

