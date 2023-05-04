# 安装依赖
python=3.7

```pip install stanza==1.2.3```

注意：stanza需要的torch版本>=1.3.0

# 下载中文模型
可以这么下载：http://nlp.stanford.edu/software/stanza/1.0.0/zh-hans/default.zip

可以去这里下载：https://huggingface.co/stanfordnlp/stanza-zh-hans/tree/main/models

这里我使用第一种下载方式：http://nlp.stanford.edu/software/stanza/1.5.0/zh-hans/default.zip

可能还需要resoureces.json，可以去这里下载：https://gitcode.net/mirrors/stanfordnlp/stanza-resources/-/blob/main/resources_1.5.0.json

# 使用代码
```python
import stanza
# print(help(stanza.Pipeline))
# model_dir指定下载模型的位置，这里我们离线下载
# stanza.download('zh', model_dir="./resources/zh-hans", verbose=False)

class CompanyExtactor:
    def __init__(self):
        # dir指定模型的位置，需要注意默认后面会加上zh-hans。
        self.zh_nlp = stanza.Pipeline('zh', processors='tokenize,ner', verbose=False, use_gpu=False, download_method=None,
                                      dir = "./resources/")

    def extract(self, text):
        zh_doc = self.zh_nlp(text)
        companies = []
        for ent in zh_doc._ents:
            if ent.type == "ORG":
                companies.append({
                    "text": ent.text,
                    "type": ent.type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
        print(companies)
        return companies

companyExtactor = CompanyExtactor()
companyExtactor.extract("山东金城医药化工股份有限公司关于持股5%以上股东减持股份的提示性公告本公司及其董事、监事、高级管理人员保证公告内容真实、准确和完整，并对公告中的虚假记载、误导性陈述或者重大遗漏承担责任。")
```
如果报错：
- 找不到resources.json，下好后放在resoureces下。
- ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with OpenSSL 1.0.2k-fips：则pip install urllib3==1.26.7
