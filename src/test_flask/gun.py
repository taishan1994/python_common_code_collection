import logging
import logging.handlers
import os
import multiprocessing
import gevent.monkey
gevent.monkey.patch_all()

bind = "0.0.0.0:9999"
chdir = "./" # gunicorn要切换到的工作目录
timeout = 60
work_class = "gevent"  # 使用gevent模式，默认是sync模式
workers = multiprocessing.cpu_count() * 2 + 1
loglevel = "info"
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s "%(a)s"'  # 设置gunicorn访问日志格式，错误日志无法设置
pidfile = "gunicorn.pid"
accesslog = "access.log"
errorlog = "error.log"
daemon = False # 是否后台运行
