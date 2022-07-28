__author__ = "Burgess Zheng"
# !/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Singleton
'''


# 实现__new__方法
# 并在将一个类的实例绑定到类变量_instance上,
# 如果cls._instance为None说明该类还没有实例化过,实例化该类,并返回
# 如果cls._instance不为None,直接返回cls._instance
class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            # cls = a = MyClass('Burgess')
            # 判断是否有a该实例存在,前面是否已经有人实例过，如果内存没有该实例...往下执行
            # 需要注明该父类的内存空间内最多允许相同名字子类的实例对象存在1个（不可多个）

            orig = super(Singleton, cls)  # farther class
            cls._instance = orig.__new__(cls)
            # orig =让cls继承指定的父类 Singleton
            # cls._instance = 创建了MyClass('Burgess') 该实例
            # 这两句相当于外面的 a= MyClass('Burgess')
        return cls._instance  # 具体的实例


class MyClass(Singleton):
    def __init__(self, name):
        self.name = name


class Nana(Singleton):
    def __init__(self, name):
        self.name = name


a = MyClass("Burgess")
print(a.name)
b = MyClass("Crystal")
print(a.name)
print(b.name)
b.name = 'xx'
print(a.name)
print(b.name)