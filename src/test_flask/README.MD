```PYTHON
pip install greenlet # 使用异步必须安装
pip install eventlet # 使用eventlet workers
pip install gevent   # 使用gevent workers

gunicorn -c gun.py server:app
```
