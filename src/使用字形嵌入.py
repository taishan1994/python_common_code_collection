# 需要到https://huggingface.co/ShannonAI/ChineseBERT-base/tree/main 下载相关文件

import numpy as np
from PIL import Image

data = np.load('STFANGSO.TTF24.npy')
arr = data[1759]
print(arr.shape)
im = Image.fromarray(arr)
if im.mode == "F":
  im = im.convert('RGB')
im.save("test.jpeg")

from typing import List

import numpy as np
import torch
from torch import nn


class GlyphEmbedding(nn.Module):
  """Glyph2Image Embedding"""

  def __init__(self, font_npy_files: List[str]):
      super(GlyphEmbedding, self).__init__()
      font_arrays = [
          np.load(np_file).astype(np.float32) for np_file in font_npy_files
      ]
      self.vocab_size = font_arrays[0].shape[0]
      self.font_num = len(font_arrays)
      self.font_size = font_arrays[0].shape[-1]
      # N, C, H, W
      font_array = np.stack(font_arrays, axis=1)
      self.embedding = nn.Embedding(
          num_embeddings=self.vocab_size,
          embedding_dim=self.font_size ** 2 * self.font_num,
          _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1]))
      )

  def forward(self, input_ids):
      """
          get glyph images for batch inputs
      Args:
          input_ids: [batch, sentence_length]
      Returns:
          images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
      """
      # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
      return self.embedding(input_ids)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/chinese-bert-wwm-ext")
text = "30%～50%患儿血清IgA浓度升高；HSP急性期血循环中表面IgA阳性的B淋巴细胞数、IgA类免疫复合物或冷球蛋白均增高；"
tokens = [i for i in text]
print(tokens)
input_ids = torch.tensor(np.array([tokenizer.convert_tokens_to_ids(tokens)])).long()
print(input_ids)

font_npy_files = ["STFANGSO.TTF24.npy", "STXINGKA.TTF24.npy", "方正古隶繁体.ttf24.npy"]
glyphEmbedding = GlyphEmbedding(font_npy_files=font_npy_files)

glyph_embedding = glyphEmbedding(input_ids)
print(glyph_embedding.shape)

"""
(24, 24)
['3', '0', '%', '～', '5', '0', '%', '患', '儿', '血', '清', 'I', 'g', 'A', '浓', '度', '升', '高', '；', 'H', 'S', 'P', '急', '性', '期', '血', '循', '环', '中', '表', '面', 'I', 'g', 'A', '阳', '性', '的', 'B', '淋', '巴', '细', '胞', '数', '、', 'I', 'g', 'A', '类', '免', '疫', '复', '合', '物', '或', '冷', '球', '蛋', '白', '均', '增', '高', '；']
tensor([[ 124,  121,  110, 8080,  126,  121,  110, 2642, 1036, 6117, 3926,  100,
          149,  100, 3849, 2428, 1285, 7770, 8039,  100,  100,  100, 2593, 2595,
         3309, 6117, 2542, 4384,  704, 6134, 7481,  100,  149,  100, 7345, 2595,
         4638,  100, 3900, 2349, 5301, 5528, 3144,  510,  100,  149,  100, 5102,
         1048, 4554, 1908, 1394, 4289, 2772, 1107, 4413, 6028, 4635, 1772, 1872,
         7770, 8039]])
torch.Size([1, 62, 1728])
"""
