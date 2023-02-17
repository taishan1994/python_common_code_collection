#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import logging.handlers #日志滚动及删除使用
 
#1.设置log日志记录格式及记录级别
#level记录级别包括DEBUG/INFO/WARNING/ERROR/CRITICAL，级别依次上升，log只会输出保存设置的级别及以上的日志。如果设置level=logging.DEBUG,则所有级别日志都会输出保存、如果level=logging.CRITICAL，则只输出保存CRITICAL级别日志
#format输出格式levelname级别名、asctime 时间、filename所在文件名、message记录内容
#datefmt 时间格式
#filename 要保存的文件名
#a写入模式，a则每次启动脚本时在原有文件中继续添加；w则每次启动脚本会重置文件然后记录
logging.basicConfig(level=logging.INFO,
                format='%(levelname)s: %(asctime)s %(filename)s %(message)s',
                datefmt='%Y-%m-%d %A %H:%M:%S',
                filename='myapp.log',
                filemode='a')
 
#2.设置log日志的标准输出打印，如果不需要在终端输出结果可忽略
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(asctime)s %(filename)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
 
#3.设置log日志文件按时间拆分记录，并保存几个历史文件，如果不需要拆分文件记录可忽略
#class logging.handlers.WatchedFileHandler(filename, mode='a', encoding=None, delay=False)
#例：设置每天保存一个log文件，以日期为后缀，保留7个旧文件。
myapp = logging.getLogger()
myapp.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(asctime)s %(filename)s %(message)s')
filehandler = logging.handlers.TimedRotatingFileHandler("myapp.log", when='d', interval=1, backupCount=7)#每 1(interval) 天(when) 重写1个文件,保留7(backupCount) 个旧文件；when还可以是Y/m/H/M/S
filehandler.suffix = "%Y-%m-%d_%H-%M-%S.log"#设置历史文件 后缀
filehandler.setFormatter(formatter)
myapp.addHandler(filehandler)
 
#4.设置log日志文件按文件大小拆分记录，并保存几个历史文件，如果不需要拆分文件记录可忽略
#class logging.handlers.RotatingFileHandler(filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=0)
myapp = logging.getLogger()
myapp.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(asctime)s %(filename)s %(message)s')
filehandler = logging.handlers. RotatingFileHandler("myapp.log", mode='a', maxBytes=1024, backupCount=2)#每 1024Bytes重写一个文件,保留2(backupCount) 个旧文件
filehandler.setFormatter(formatter)
myapp.addHandler(filehandler)
 
 
#使用
logging.debug('debug message : %s , result: %s',info,result)  
logging.info('info message')  
logging.warning('warning message')  
logging.error('error message')  
logging.critical('critical message') 
