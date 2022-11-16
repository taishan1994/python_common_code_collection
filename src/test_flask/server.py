import time
from flask import Flask
app = Flask(__name__)


@app.route("/")
def hello_flask():
    time.sleep(30)
    return "hello world"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9999, debug=False, threaded=False)
