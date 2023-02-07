# pip install pypinyin
# 需要去https://huggingface.co/ShannonAI/ChineseBERT-base/tree/main/config 下载相关文件


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/chinese-bert-wwm-ext")
text = "30%～50%患儿血清IgA浓度升高；HSP急性期血循环中表面IgA阳性的B淋巴细胞数、IgA类免疫复合物或冷球蛋白均增高；"
tokens = [i for i in text]
print(tokens)

import os
import json
from pypinyin import pinyin, Style
config_path = "./"

with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
  pinyin_dict = json.load(fin)
# load char id map tensor
with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
  id2pinyin = json.load(fin)
# load pinyin map tensor
with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
  pinyin2tensor = json.load(fin)

pinyin_list = pinyin(text, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
print(pinyin_list)
pinyin_locs = {}
# get pinyin of each location
for index, item in enumerate(pinyin_list):
  pinyin_string = item[0]
  # not a Chinese character, pass
  if pinyin_string == "not chinese":
    continue
  if pinyin_string in pinyin2tensor:
    pinyin_locs[index] = pinyin2tensor[pinyin_string]
  else:
    ids = [0] * 8
    for i, p in enumerate(pinyin_string):
      if p not in pinyin_dict["char2idx"]:
        ids = [0] * 8
        break
      ids[i] = pinyin_dict["char2idx"][p]
    pinyin_locs[index] = ids

# find chinese character location, and generate pinyin ids
pinyin_ids = []
for idx in range(len(tokens)):
  if idx in pinyin_locs:
    pinyin_ids.append(pinyin_locs[idx])
  else:
    pinyin_ids.append([0] * 8)

print(pinyin_locs)
print(pinyin_ids)

import torch
import numpy as np
pinyin_ids = torch.tensor(np.array([pinyin_ids])).long()
print(pinyin_ids.shape)

from torch import nn
from torch.nn import functional as F


class PinyinEmbedding(nn.Module):
  def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path):
      """
          Pinyin Embedding Module
      Args:
          embedding_size: the size of each embedding vector
          pinyin_out_dim: kernel number of conv
      """
      super(PinyinEmbedding, self).__init__()
      with open(os.path.join(config_path, 'pinyin_map.json')) as fin:
          pinyin_dict = json.load(fin)
      self.pinyin_out_dim = pinyin_out_dim
      self.embedding = nn.Embedding(len(pinyin_dict['idx2char']), embedding_size)
      self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                            stride=1, padding=0)

  def forward(self, pinyin_ids):
      """
      Args:
          pinyin_ids: (bs*sentence_length*pinyin_locs)
      Returns:
          pinyin_embed: (bs,sentence_length,pinyin_out_dim)
      """
      # input pinyin ids for 1-D conv
      embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
      bs, sentence_length, pinyin_locs, embed_size = embed.shape
      view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
      input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
      # conv + max_pooling
      pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
      pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
      return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]

pinyinEmbedding = PinyinEmbedding(embedding_size=300, pinyin_out_dim=128, config_path="./")
pinyin_embedding = pinyinEmbedding(pinyin_ids)
print(pinyin_embedding.shape)


"""
['3', '0', '%', '～', '5', '0', '%', '患', '儿', '血', '清', 'I', 'g', 'A', '浓', '度', '升', '高', '；', 'H', 'S', 'P', '急', '性', '期', '血', '循', '环', '中', '表', '面', 'I', 'g', 'A', '阳', '性', '的', 'B', '淋', '巴', '细', '胞', '数', '、', 'I', 'g', 'A', '类', '免', '疫', '复', '合', '物', '或', '冷', '球', '蛋', '白', '均', '增', '高', '；']
[['not chinese'], ['not chinese'], ['not chinese'], ['not chinese'], ['not chinese'], ['not chinese'], ['not chinese'], ['huan4'], ['er2', 'er', 'ren2'], ['xue4'], ['qing1'], ['not chinese'], ['not chinese'], ['not chinese'], ['nong2'], ['du4'], ['sheng1'], ['gao1', 'gao4'], ['not chinese'], ['not chinese'], ['not chinese'], ['not chinese'], ['ji2'], ['xing4'], ['qi1', 'ji1'], ['xue4'], ['xun2'], ['huan2'], ['zhong1'], ['biao3'], ['mian4'], ['not chinese'], ['not chinese'], ['not chinese'], ['yang2'], ['xing4'], ['de', 'di1', 'di2', 'di4'], ['not chinese'], ['lin2', 'lin4'], ['ba1'], ['xi4'], ['bao1', 'pao2', 'pao4'], ['shu4', 'shu3', 'shuo4'], ['not chinese'], ['not chinese'], ['not chinese'], ['not chinese'], ['lei4'], ['mian3', 'wen4', 'wan3'], ['yi4'], ['fu4'], ['he2'], ['wu4'], ['huo4', 'yu4'], ['leng3', 'ling2', 'ling3'], ['qiu2'], ['dan4'], ['bai2', 'bo2'], ['jun1', 'yun4'], ['zeng1', 'zeng4', 'ceng2'], ['gao1', 'gao4'], ['not chinese']]
{7: [13, 26, 6, 19, 4, 0, 0, 0], 8: [10, 23, 2, 0, 0, 0, 0, 0], 9: [29, 26, 10, 4, 0, 0, 0, 0], 10: [22, 14, 19, 12, 1, 0, 0, 0], 14: [19, 20, 19, 12, 2, 0, 0, 0], 15: [9, 26, 4, 0, 0, 0, 0, 0], 16: [24, 13, 10, 19, 12, 1, 0, 0], 17: [12, 6, 20, 1, 0, 0, 0, 0], 22: [15, 14, 2, 0, 0, 0, 0, 0], 23: [29, 14, 19, 12, 4, 0, 0, 0], 24: [22, 14, 1, 0, 0, 0, 0, 0], 25: [29, 26, 10, 4, 0, 0, 0, 0], 26: [29, 26, 19, 2, 0, 0, 0, 0], 27: [13, 26, 6, 19, 2, 0, 0, 0], 28: [31, 13, 20, 19, 12, 1, 0, 0], 29: [7, 14, 6, 20, 3, 0, 0, 0], 30: [18, 14, 6, 19, 4, 0, 0, 0], 34: [30, 6, 19, 12, 2, 0, 0, 0], 35: [29, 14, 19, 12, 4, 0, 0, 0], 36: [9, 10, 5, 0, 0, 0, 0, 0], 38: [17, 14, 19, 2, 0, 0, 0, 0], 39: [7, 6, 1, 0, 0, 0, 0, 0], 40: [29, 14, 4, 0, 0, 0, 0, 0], 41: [7, 6, 20, 1, 0, 0, 0, 0], 42: [24, 13, 26, 4, 0, 0, 0, 0], 47: [17, 10, 14, 4, 0, 0, 0, 0], 48: [18, 14, 6, 19, 3, 0, 0, 0], 49: [30, 14, 4, 0, 0, 0, 0, 0], 50: [11, 26, 4, 0, 0, 0, 0, 0], 51: [13, 10, 2, 0, 0, 0, 0, 0], 52: [28, 26, 4, 0, 0, 0, 0, 0], 53: [13, 26, 20, 4, 0, 0, 0, 0], 54: [17, 10, 19, 12, 3, 0, 0, 0], 55: [22, 14, 26, 2, 0, 0, 0, 0], 56: [9, 6, 19, 4, 0, 0, 0, 0], 57: [7, 6, 14, 2, 0, 0, 0, 0], 58: [15, 26, 19, 1, 0, 0, 0, 0], 59: [31, 10, 19, 12, 1, 0, 0, 0], 60: [12, 6, 20, 1, 0, 0, 0, 0]}
[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [13, 26, 6, 19, 4, 0, 0, 0], [10, 23, 2, 0, 0, 0, 0, 0], [29, 26, 10, 4, 0, 0, 0, 0], [22, 14, 19, 12, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [19, 20, 19, 12, 2, 0, 0, 0], [9, 26, 4, 0, 0, 0, 0, 0], [24, 13, 10, 19, 12, 1, 0, 0], [12, 6, 20, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [15, 14, 2, 0, 0, 0, 0, 0], [29, 14, 19, 12, 4, 0, 0, 0], [22, 14, 1, 0, 0, 0, 0, 0], [29, 26, 10, 4, 0, 0, 0, 0], [29, 26, 19, 2, 0, 0, 0, 0], [13, 26, 6, 19, 2, 0, 0, 0], [31, 13, 20, 19, 12, 1, 0, 0], [7, 14, 6, 20, 3, 0, 0, 0], [18, 14, 6, 19, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [30, 6, 19, 12, 2, 0, 0, 0], [29, 14, 19, 12, 4, 0, 0, 0], [9, 10, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [17, 14, 19, 2, 0, 0, 0, 0], [7, 6, 1, 0, 0, 0, 0, 0], [29, 14, 4, 0, 0, 0, 0, 0], [7, 6, 20, 1, 0, 0, 0, 0], [24, 13, 26, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [17, 10, 14, 4, 0, 0, 0, 0], [18, 14, 6, 19, 3, 0, 0, 0], [30, 14, 4, 0, 0, 0, 0, 0], [11, 26, 4, 0, 0, 0, 0, 0], [13, 10, 2, 0, 0, 0, 0, 0], [28, 26, 4, 0, 0, 0, 0, 0], [13, 26, 20, 4, 0, 0, 0, 0], [17, 10, 19, 12, 3, 0, 0, 0], [22, 14, 26, 2, 0, 0, 0, 0], [9, 6, 19, 4, 0, 0, 0, 0], [7, 6, 14, 2, 0, 0, 0, 0], [15, 26, 19, 1, 0, 0, 0, 0], [31, 10, 19, 12, 1, 0, 0, 0], [12, 6, 20, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
torch.Size([1, 62, 8])
torch.Size([1, 62, 128])
"""

