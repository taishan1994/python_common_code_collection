import argparse
import json
from flask import Flask, request, jsonify
import torch
from transformers import (
  AutoModelForTokenClassification,
  AutoTokenizer,
  pipeline,
)
from service_streamer import ThreadedStreamer, Streamer
from bert_model import TextInfillingModel, ManagedBertModel

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--host", help="ip地址", type=str)
parser.add_argument("--port", help="端口", type=str)
parser.add_argument("--use_service_streamer", help="是否使用service_streamer", action="store_true")
args = parser.parse_args()

@app.route("/fill_mask", methods=["POST"])
def get_ner_result():
  inputs = request.get_json()
  text = inputs["text"]
  result = model.predict(text)
  return json.dumps(result, ensure_ascii=False)
	
if __name__ == "__main__":
  ori_model = TextInfillingModel()
  if args.use_service_streamer:
    print("使用service_streamer!!!")
    # model = ThreadedStreamer(ori_model.predict, batch_size=32, max_latency=1)
    # model = Streamer(ori_model.predict, batch_size=32, max_latency=1, worker_num=4) # 要在main里面初始化
    model = Streamer(ManagedBertModel, 32, 0.1, worker_num=2, cuda_devices=(0,))
  else:
    model = ori_model
  app.run(host=args.host, port=args.port, threaded=True)
