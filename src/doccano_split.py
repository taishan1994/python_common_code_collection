# 对doccano标注好的数据进行分句


# coding=utf-8
import json
from pprint import pprint
import re


def cut_sentences_v1(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")


def cut_sentences_v2(sent):
    """
    the second rank of spilt sentence, split '；' | ';'
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
    return sent.split("\n")


def cut_sentences_v3(sent):
    """以逗号进行分句"""
    sent = re.sub('([,，])([^”’])', r'\1\n\2', sent)
    return sent.split("\n")


def cut_sentences_main(text, max_seq_len):
    # 将句子分句，细粒度分句后再重新合并
    sentences = []

    # 细粒度划分
    sentences_v1 = cut_sentences_v1(text)
    # print("sentences_v1=", sentences_v1)
    for sent_v1 in sentences_v1:
        # print(sent_v1)
        if len(sent_v1) > max_seq_len:
            sentences_v2 = cut_sentences_v2(sent_v1)
            sentences.extend(sentences_v2)
        else:
            sentences.append(sent_v1)
    # if ''.join(sentences) != text:
        # print(len(''.join(sentences)), len(text))

    res = []
    for sent in sentences:
        # print(sentences)
        if len(sent) > max_seq_len:
            sent_v3 = cut_sentences_v3(sent)
            # print(sent_v3)
            tmp = []
            length = 0
            for i in range(len(sent_v3)):
                if length + len(sent_v3[i]) < max_seq_len:
                    tmp.append(sent_v3[i])
                    length = length + len(sent_v3[i])
                else:
                    if "".join(tmp) != "":
                        res.append("".join(tmp))
                        tmp = [sent_v3[i]]
                        length = len(sent_v3[i])
            if "".join(tmp) != "":
                res.append("".join(tmp))
        else:
            res.append(sent)
    # assert ''.join(sentences) == text
    # 过滤掉空字符
    final_res = []
    for i in res:
        if i.strip() != "":
            final_res.append(i)
    return final_res


def refactor(text):
    # pprint(text)
    content = text["text"]
    # print(json.dumps(text, ensure_ascii=False))
    cut_text = cut_sentences_main(content, 512)
    assert len(content) == len("".join(cut_text))
    entities = text["entities"]
    entities = sorted(entities, key=lambda x: x["start_offset"])
    relations = text["relations"]
    init_len = 0
    content_id = text["id"]

    # for entity in entities:
    #     eid = entity["id"]
    #     label = entity["label"]
    #     start = entity["start_offset"]
    #     end = entity["end_offset"]
    #     ent = content[start:end]
    #     print(eid, label, ent)
    
    res = []
    for i, tex in enumerate(cut_text):
        # print(len(tex), tex)
        tmp = {}
        tmp["id"] = str(content_id) + "_{}".format(i)
        tmp["text"] = tex
        tmp["entities"] = []
        tmp["relations"] = []
        for entity in entities:
            eid = entity["id"]
            label = entity["label"]
            start = entity["start_offset"]
            end = entity["end_offset"]
            ent = content[start:end]
            # print(ent)
            if init_len <= end <= init_len + len(tex):
                start = start - init_len
                end = end - init_len
                tmp["entities"].append({
                    "id": eid,
                    "label": label,
                    "start_offset": start,
                    "end_offset": end,
                    "ent": tex[start:end],
                })
            elif end < init_len:
                continue
            elif end > init_len:
                break
        init_len = init_len + len(tex)
        ent_ids = [ent["id"] for ent in tmp["entities"]]
        for rel in relations:
            from_id = rel["from_id"]
            to_id = rel["to_id"]
            if from_id in ent_ids and to_id in ent_ids:
                tmp["relations"].append(rel)
        # print(init_len)
        # pprint(tmp)
        res.append(tmp)
    return res

with open("admin.jsonl", "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")

f_res = []
for d in data:
    d = json.loads(d)
    f_res.append(d)
    tmp = refactor(d)
    f_res.extend(tmp)

with open("output.jsonl", "w", encoding="utf-8") as fp:
    fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in f_res]))

