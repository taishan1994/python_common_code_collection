# 前言
我们经常可以看到使用argparse来接收命令行的参数，一般的这么使用：
新建一个test.py
```python
import argparse

parser = argparse.ArgumentParser()
# type是要传入的参数的数据类型  help是该参数的提示信息, default是默认值
parser.add_argument('--model_name', default="bert", type=str, help='模型名称')

args = parser.parse_args()
print(args.model_name)
```
如果我们直接运行`python test.py`，会得到bert。我们可以在运行时传入定义的参数来改变其中的值，比如`python test.py --model_name="roberta"`或者`python test.py --model_name "roberta"`。需要注意的是，字符串要使用双引号，而不能使用单引号。

当我们要定义的参数比较多的时候，每一个参数都需要写一个add_argument，这样**既麻烦**，而且**查看某个参数的值得时候不是很直观**。

# 自定义参数解析
有时候为了直观，会把参数直接用字典或者类来存储参数，但每次需要改动参数的时候都要找到相应的文件修改，也挺麻烦。那么，有没有一种方法**既结合字典的直观，又有argparse的灵活性**呢？

我们知道，另一种接收命令行传过来的参数的方法是使用`sys`，比如：
```python
import sys
args = sys.argv
args = args[1:]
print(args)
```
输入：`python test.py --model_name="roberta"`会得到：`['--model_name=roberta']`，也就是python test.py后面的参数会以列表的形式返回。需要注意参数值需要用=赋值，而不能有空格了。接下来就是我们的主代码：
```python
import re
import ast
import sys
from pprint import pprint


class ConfigParser:
    def __init__(self, config):
        self.config = config
        assert isinstance(config, dict)
        args = sys.argv
        args = args[1:]
        self.args = args

    def judge_type(self, value):
        """利用正则判断参数的类型"""
        if value.isdigit():
            return int(value)
        elif re.match(r'^-?\d+\.?\d*$', value):
            return float(value)
        elif value.lower() in ["true", "false"]:
            return True if value == "true" else False
        else:
            try:
                st = ast.literal_eval(value)
                return st
            except Exception as e:
                return value

    def get_args(self):
        return_args = {}
        for arg in self.args:
            arg = arg.split("=")
            arg_name, arg_value = arg
            if "--" in arg_name:
                arg_name = arg_name.split("--")[1]
            elif "-" in arg_name:
                arg_name = arg_name.split("-")[1]
            return_args[arg_name] = self.judge_type(arg_value)
        return return_args

    # 定义一个函数，用于递归获取字典的键
    def get_dict_keys(self, config, prefix=""):
        result = {}
        for k, v in config.items():
            new_key = prefix + "_" + k if prefix else k
            if isinstance(v, dict):
                result.update(self.get_dict_keys(v, new_key))
            else:
                result[new_key] = v
        return result

    # 定义一个函数，用于将嵌套字典转换为类的属性
    def dict_to_obj(self, merge_config):
        # 如果d是字典类型，则创建一个空类
        if isinstance(merge_config, dict):
            obj = type("", (), {})()
            # 将字典的键转换为类的属性，并将字典的值递归地转换为类的属性
            for k, v in merge_config.items():
                setattr(obj, k, self.dict_to_obj(v))
            return obj
        # 如果d不是字典类型，则直接返回d
        else:
            return merge_config

    def set_args(self, args, cls):
        """遍历赋值"""
        for key, value in args.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise Exception(f"参数【{key}】不在配置中，请检查！")
        return cls

    def parse_main(self):
        # 获取命令行输入的参数
        cmd_args = self.get_args()
        # 合并字典的键，用_进行连接
        merge_config = self.get_dict_keys(self.config)
        # 将字典配置转换为类可调用的方式
        class_config = self.dict_to_obj(merge_config)
        # 合并命令行参数到类中
        cls = self.set_args(cmd_args, class_config)
        return cls


if __name__ == '__main__':
    config = {
        "data_name": "msra",
        "ouput_dir": "./checkpoint/",
        "model_name": "bert",
        "do_predict": True,
        "do_eval": True,
        "do_test": True,
        "max_seq_len": 512,
        "lr_steps": [80, 180],
        "optimizer": {
            "adam": {
                "leraning_rate": 5e-3,
            },
            "adamw": {
                "leraning_rate": 5e-3,
            }
        }
    }

    print("最初参数：")
    pprint(config)
    config_parser = ConfigParser(config)
    args = config_parser.parse_main()
    print("修改后参数：")
    print("="*100)
    pprint(vars(args))

    print("="*100)
    print(args.model_name)
    print(args.optimizer_adam_leraning_rate)
```
说明一下主要的步骤：
- 1、我们将配置定义在字典中。
- 2、通过sys接收命令行的参数，并对参数的值进行解析：整型、字符串、浮点型、布尔型、列表、字典等。然后处理为字典。
- 3、解析配置中的键，将同一层的键用_拼接起来并重新赋值。
- 4、将配置转换为类可调用的形式。
- 5、将命令行参数赋值给配置里面的参数，如果不匹配则抛出异常。

使用的时候我们只需要定义好字典配置，然后在命令行传入参数就可以修改配置了。需要注意的是：
- 1、命令行参数要用=进行赋值。
- 2、对于嵌套的字典配置，使用类的方式调用的时候需要使用_进行分隔。
- 3、参数值里面不能包含多的空格。比如[20,40]不能写为[20, 40]，否则会报错。或者"[20, 40]"也行。

最后我们来用用：
- 1、使用字典定义配置。
- 2、初始化ConfigParser并传入配置。
- 3、调用config_parser.parse_main()得到解析之后的配置，就可以使用了。

原始参数：
```python
{'data_name': 'msra',
 'do_eval': True,
 'do_predict': True,
 'do_test': True,
 'lr_steps': [80, 180],
 'max_seq_len': 512,
 'model_name': 'bert',
 'optimizer': {'adam': {'leraning_rate': 0.005},
               'adamw': {'leraning_rate': 0.005}},
 'ouput_dir': './checkpoint/'}
```
我们不作任何修改，直接运行`python test.py`，得到解析后的参数：
```python
{'data_name': 'msra',
 'do_eval': True,
 'do_predict': True,
 'do_test': True,
 'lr_steps': [80, 180],
 'max_seq_len': 512,
 'model_name': 'bert',
 'optimizer_adam_leraning_rate': 0.005,
 'optimizer_adamw_leraning_rate': 0.005,
 'ouput_dir': './checkpoint/'}
```
最后我们通过命令行修改参数：`python test.py --model_name=roberta --lr_steps=[20, 40] --do_predict=False --optimizer_adam_leraning_rate=0.01`得到：
```python
{'data_name': 'msra',
 'do_eval': True,
 'do_predict': False,
 'do_test': True,
 'lr_steps': [20, 40],
 'max_seq_len': 512,
 'model_name': 'roberta',
 'optimizer_adam_leraning_rate': 0.01,
 'optimizer_adamw_leraning_rate': 0.005,
 'ouput_dir': './checkpoint/'}
```
解析完成后可以使用 **.** 来获取具体属性的值。
# 总结
通过以上形式，我们可以：
- 将参数配置直接定义到字典中，更直观。
- 可以通过命令行灵活修改参数配置。
- 可以在命令行里面直接传入布尔值、列表、字典等。
