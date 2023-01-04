
import argparse
import json
from flask import Flask, request, jsonify
import torch
from transformers import (
  AutoModelForTokenClassification,
  AutoTokenizer,
  pipeline,
)

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--host", help="ip地址", type=str)
parser.add_argument("--port", help="端口", type=str)
args = parser.parse_args()

class NerModel:
  def __init__(self, device):
    model = AutoModelForTokenClassification.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
    tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
    self.ner = pipeline('ner', model=model, tokenizer=tokenizer, device=device)

  def get_entities(self, result):
    ents = []
    for res in result:
      ent = []
      ent_text = []
      i = 0
      while i < len(res):
        if "B-" in res[i]["entity"]:
          label = res[i]["entity"].split("-")[-1]
          start = res[i]["start"]
          ent_text.append(res[i]["word"])
          end = res[i]["end"]
          i += 1
          while i < len(res) and "I-" in res[i]["entity"] and res[i]["entity"].split("-")[-1] == label:
            ent_text.append(res[i]["word"])
            end = res[i]["end"]
            i += 1
          ent.append({"text":"".join(ent_text), "start":str(start), "end":str(end), "label":label})
          ent_text = []
      ents.append(ent)
    return ents

  def predict(self, batch):
    if isinstance(batch, str):
      batch = [batch]
      result = self.ner(batch)
      result = [result]
    else:
      result = self.ner(batch)
   
    entities = self.get_entities(result)
    return {"result":[(text, ent) for text, ent in zip(batch, entities)]}

device = 0 if torch.cuda.is_available() else -1
nerModel = NerModel(device)

@app.route("/ner", methods=["POST"])
def get_ner_result():
  inputs = request.get_json()
  text = inputs["text"]
  result = nerModel.predict(text)
  return json.dumps(result, ensure_ascii=False)
	
if __name__ == "__main__":
  app.run(host=args.host, port=args.port, threaded=True)
