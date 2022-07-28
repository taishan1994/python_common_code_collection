__author__ = "Burgess Zheng"
# !/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Flyweight
'''


class FlyweightBase(object):
    _instances = dict()  # 皴法实例化的对象内存地址

    def __init__(self, *args, **kwargs):
        # 继承的子类必须初始化
        raise NotImplementedError

    def __new__(cls, *args, **kwargs):
        print(cls._instances, type(cls))  # cls 就是你要实例化的子类如：obj = Spam(1,abc)
        return cls._instances.setdefault(
            (cls, args, tuple(kwargs.items())),  # key   （实例和参数）obj = Spam(y,x)
            super(FlyweightBase, cls).__new__(cls)  # value  #实例化新的对象的内存地址
            # 调用自身的_instances字典，如果没有往父类找_instances字典
            # setdefault：判断_instances字典是否有该key:obj = Spam(y,x)实例 ,
            #               如果有，返回该key的value（上次实例化对象（内存地址））
            # setdefault： 如果找不到key：obj = Spam(y,x)实例 ，就在_instances字典就创建该key，value为新实例化对象（内存地址）
            #               返回该新创建key的value(该次实例化的对象（内存地址）
            # 这也就说明你实例化对象的时候，如果形参相同的话，不用实例化，直接返回已存在的实例的内存）
        )


class Spam(FlyweightBase):
    '''精子类'''

    def test_data(self):
        pass

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def test_data(self):
        print("精子准备好了", self.a, self.b)


class Egg(FlyweightBase):
    '''卵类'''

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def test_data(self):
        print("卵子准备好了", self.x, self.y)


spam1 = Spam(1, 'abc')
spam2 = Spam(1, 'abc')
spam3 = Spam(3, 'DEF')

egg1 = Egg(1, 'abc')
print(id(spam1), id(spam2), id(spam3))

# egg2 = Egg(4,'abc')
# assert spam1 is spam2
# assert egg1 is not spam1
# print(id(spam1),id(spam2))
# spam2.test_data()
# egg1.test_data()
# print(egg1._instances)
# print(egg1._instances.keys())