import sys
import traceback

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import jieba
from loguru import logger

logger.add("test.log", enqueue=True)

app = FastAPI()


@app.get('/test/name={name}')
def test_get(name):
    c = "你好！" + name
    res = {"res": c}
    return res


@app.get('/')
def index():
    message = "欢迎来到主页面"
    return message


def seg(text):
    return jieba.lcut(text, cut_all=False)


@app.post('/seg')
def test_post(text: str):
    try:
        seg_text = seg(text)
        status_code = 200
        message = "成功"
        data = seg_text
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logger.error("".join(lines))
        status_code = 500
        message = "错误！{}".format(e)
        data = []
    res = {"code": status_code, "message": message, "data": data}
    return JSONResponse(content=res, status_code=status_code)


if __name__ == '__main__':
    uvicorn.run(app="test:app",
                host="0.0.0.0",
                port=8080,
                workers=2,
                reload=True,
                debug=True)
