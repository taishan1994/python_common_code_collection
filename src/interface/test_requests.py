import requests

text = '恶魔猎手吧-百度贴吧--《魔兽世界》恶魔猎手职业贴吧...'
params = {
  "text": text,
}
url = 'http://127.0.0.1:5000/ner'
result = requests.post(url, json=params)
result = result.text
print(result)