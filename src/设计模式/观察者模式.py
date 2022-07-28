__author__ = "Burgess Zheng"
# !/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Observer
'''


# 观察者（Observer）模式又名发布-订阅（Publish/Subscribe）模式
# 当我们希望一个对象的状态发生变化，那么依赖与它的所有对象都能相应变化(获得通知),那么就可以用到Observer模式， 其中的这些依赖对象就是观察者的对象，那个要发生变化的对象就是所谓’观察者’

class ObserverBase(object):
    '''观察者基类'''  # 放哨者

    def __init__(self):
        self._observerd_list = []  # 被通知对象

    def attach(self, observe_subject):
        '''
        添加要观察的对象
        :param observe_subject:
        :return:
        '''
        if observe_subject not in self._observerd_list:
            self._observerd_list.append(observe_subject)
            print("[%s]已经将[%s]加入观察队列..." % (self.name, observe_subject))

    def detach(self, observe_subject):
        '''
        解除观察关系
        :param observe_subject:
        :return:
        '''
        try:
            self._observerd_list.remove(observe_subject)
            print("不再观察[%s]" % observe_subject)
        except ValueError:
            pass

    def notify(self):
        '''
        通知所有被观察者
        :return:
        '''
        for objserver in self._observerd_list:
            objserver.update(self)


class Observer(ObserverBase):
    '''观察者类'''

    def __init__(self, name):
        super(Observer, self).__init__()
        self.name = name
        self._msg = ''

    @property  # 外部执行d.eat 去掉括号
    def msg(self):
        '''
        当前状况
        :return:
        '''
        return self._msg

    @msg.setter  # 设置属性（一个方法变成一个静态的属性）
    def msg(self, content):
        self._msg = content
        self.notify()


# 目前结论：一个方法变成一个静态的属性 （装饰了@property）
# 通过另外一个相同名字的方法（装饰@msg.setter ）进行修改该属性
# 简单： d = Observer('xxx')
# print(d.msg)  结果：空
# d.msg = 'xxx'
# #这种形式调用该对象有装饰setter的方法，xxx作为形参
# print(d.msg)  结果：xxx


class GCDViewer(object):
    '''
    共军被观察者
    '''

    def update(self, observer_subject):
        print("共军:收到[%s]消息[%s] " % (observer_subject.name, observer_subject.msg))


class GMDViewer(object):
    '''
    国军被观察者
    '''

    def update(self, observer_subject):
        print("国军:收到[%s]消息[%s] " % (observer_subject.name, observer_subject.msg))


if __name__ == "__main__":
    observer1 = Observer("共军放哨者")
    observer2 = Observer("国军放哨者")

    gongjun1 = GCDViewer()
    guojun1 = GMDViewer()

    observer1.attach(gongjun1)
    observer1.attach(guojun1)

    observer2.attach(guojun1)
    # 里面的每个对象都会响应
    observer1.msg = "\033[32;1m敌人来了...\033[0m"

    observer2.msg = "\033[31;1m前方发现敌人,请紧急撤离,不要告诉共军\033[0m"