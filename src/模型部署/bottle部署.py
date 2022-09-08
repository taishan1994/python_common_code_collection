"""
pip install gevent
"""
from gevent import monkey

monkey.patch_all()

import json
import jieba
from bottle import route, run, request


def json_dumps(data, msg="请求成功", code=200):
    return json.dumps({
        "msg": msg,
        "code": code,
        "data": data
    }, ensure_ascii=False)


@route('/stream', method="POST")
def stream():
    text = request.json['text']
    try:
        data = jieba.lcut(text, cut_all=False)
    except Exception as e:
        return json_dumps(data=[], msg=str(e), code=500)
    return json_dumps(data=data)


run(host='0.0.0.0', port=8080, server='gevent', debug=True)
