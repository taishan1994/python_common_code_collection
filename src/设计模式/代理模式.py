__author__ = "Burgess Zheng"
# !/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Proxy
'''


# 代理模式
# 应用特性：需要在通信双方中间需要一些特殊的中间操作时引用，多加一个中间控制层。
# 结构特性：建立一个中间类，创建一个对象，接收一个对象，然后把两者联通起来

class sender_base:
    def __init__(self):
        pass

    def send_something(self, something):
        pass


class send_class(sender_base):
    def __init__(self, receiver):
        self.receiver = receiver

    def send_something(self, something):
        print("SEND " + something + ' TO ' + self.receiver.name)


class agent_class(sender_base):
    def __init__(self, receiver):
        self.send_obj = send_class(receiver)

    def send_something(self, something):
        self.send_obj.send_something(something)


class receive_class:
    def __init__(self, someone):
        self.name = someone


if '__main__' == __name__:
    receiver = receive_class('Burgess')
    agent = agent_class(receiver)
    agent.send_something('agentinfo')

    print(receiver.__class__)
    print(agent.__class__)