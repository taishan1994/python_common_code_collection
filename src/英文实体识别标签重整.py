from transformers import BertTokenizer, BertModel

eng_tokenizer = BertTokenizer.from_pretrained("model_hub/bert-base-cased", do_lower_case=False)

text1 = ['West', 'Indian', 'all-rounder', 'Phil', 'Simmons', 'took', 'four', 'for', '38', 'on', 'Friday', 'as',
         'Leicestershire', 'beat', 'Somerset', 'by', 'an', 'innings', 'and', '39', 'runs', 'in', 'two', 'days', 'to',
         'take', 'over', 'at', 'the', 'head', 'of', 'the', 'county', 'championship', '.']
example = {"id": 64,
           "text": ["June", "25-27", "v", "British", "Universities", "(", "at", "Oxford", ",", "three", "days", ")"],
           "labels": ["O", "O", "O", "B-ORG", "I-ORG", "O", "O", "B-LOC", "O", "O", "O", "O"]}
# example = {"id": 351, "text": ["the", "$", "1.2", "million", "Greater", "Milwaukee", "Open", "at", "the", "par-71", ","], "labels": ["O", "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "O", "O", "O"]}
# example = {"id": 37, "text": ["AL-AIN", ",", "United", "Arab", "Emirates", "1996-12-06"],"labels": ["B-LOC", "O", "B-LOC", "I-LOC", "I-LOC", "O"]}
label = example["labels"]
text1 = example["text"]
text2 = eng_tokenizer.tokenize(" ".join(text1))
print(text1)
print(text2)
print(label)


def conver_labels_to_biolabels(labels):
    label2id = {"O": 0}
    id2label = {0: "O"}
    i = 1
    for label in labels:
        tmp_label = "B-" + label
        label2id[tmp_label] = i
        id2label[i] = tmp_label
        i += 1
        tmp_label = "I-" + label
        label2id[tmp_label] = i
        id2label[i] = tmp_label
        i += 1
    return label2id, id2label


def align_label_example(ori_input, tokenized_input, label, label_all_tokens=False):
    """这里目前label_all_tokens只能设置为True"""
    i, j = 0, 0
    ids = []
    # 这里需要遍历tokenizer后的列表
    while i < len(tokenized_input):
        if tokenized_input[i] == ori_input[j]:
            ids.append(label2id[label[j]])
            # ids.append(label[j])
            i += 1
            j += 1
        else:
            tmp = []
            tmp.append(tokenized_input[i])  # 将当前的加入的tmp
            ids.append(label2id[label[j]])  # 当前的id加入到ids
            i += 1
            while i < len(tokenized_input) and "".join(tmp) != ori_input[j]:
                ori_word = tokenized_input[i]
                if ori_word[:2] == "##":
                    tmp.append(ori_word[2:])
                    if label[j] == "O":
                        ids.append(label2id[label[j]])
                    else:
                        ids.append(label2id["I-" + label[j].split("-")[-1]] if label_all_tokens else -100)
                else:
                    if label[j] == "O":
                        ids.append(label2id[label[j]])
                    else:
                        if "O" == label[j]:
                            ids.append(label2id[label[j]])
                        else:
                            ids.append(label2id["I-" + label[j].split("-")[-1]])
                    tmp.append(ori_word)
                    # ids.append(label[j])
                i += 1
            j += 1
    assert len(ids) == len(tokenized_input)
    return ids


with open("data/conll2003/mid_data/labels.txt", "r") as fp:
    labels = fp.read().strip().split("\n")
label2id, id2label = conver_labels_to_biolabels(labels)
print(align_label_example(text1, text2, label, label_all_tokens=False))

"""
D:\software\anaconda3\envs\keras_classification\python.exe D:/flow/CAIL/ner/pytorch_bert_english_ner/main.py
['June', '25-27', 'v', 'British', 'Universities', '(', 'at', 'Oxford', ',', 'three', 'days', ')']
['June', '25', '-', '27', 'v', 'British', 'Universities', '(', 'at', 'Oxford', ',', 'three', 'days', ')']
['O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O']
[0, 0, 0, 0, 0, 7, 8, 0, 0, 1, 0, 0, 0, 0]

"""

# 方法二
from transformers import BertTokenizer, BertModel, PreTrainedTokenizerFast
from tokenizers import BertWordPieceTokenizer

tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)

examples = [
  {"id": 25, "text": ["Italy", "recalled", "Marcello", "Cuttitta"], "labels": ["B-LOC", "O", "B-PER", "I-PER"]},
  {"id": 26, "text": ["on", "Friday", "for", "their", "friendly", "against", "Scotland", "at", "Murrayfield", "more", "than", "a", "year", "after", "the", "30-year-old", "wing", "announced", "he", "was", "retiring", "following", "differences", "over", "selection", "."], "labels": ["O", "O", "O", "O", "O", "O", "B-LOC", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
  {"id": 27, "text": ["Cuttitta", ",", "who", "trainer", "George", "Coste", "said", "was", "certain", "to", "play", "on", "Saturday", "week", ",", "was", "named", "in", "a", "21-man", "squad", "lacking", "only", "two", "of", "the", "team", "beaten", "54-21", "by", "England", "at", "Twickenham", "last", "month", "."], "labels": ["B-PER", "O", "O", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "B-LOC", "O", "O", "O"]},
  {"id": 28, "text": ["Stefano", "Bordon", "is", "out", "through", "illness", "and", "Coste", "said", "he", "had", "dropped", "back", "row", "Corrado", "Covi", ",", "who", "had", "been", "recalled", "for", "the", "England", "game", "after", "five", "years", "out", "of", "the", "national", "team", "."], "labels": ["B-PER", "I-PER", "O", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O", "O", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
  {"id": 29, "text": ["Cuttitta", "announced", "his", "retirement", "after", "the", "1995", "World", "Cup", ",", "where", "he", "took", "issue", "with", "being", "dropped", "from", "the", "Italy", "side", "that", "faced", "England", "in", "the", "pool", "stages", "."], "labels": ["B-PER", "O", "O", "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O"]},
  {"id": 30, "text": ["Coste", "said", "he", "had", "approached", "the", "player", "two", "months", "ago", "about", "a", "comeback", "."], "labels": ["B-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
  {"id": 31, "text": ["\"", "He", "ended", "the", "World", "Cup", "on", "the", "wrong", "note", ",", "\"", "Coste", "said", "."], "labels": ["O", "O", "O", "O", "B-MISC", "I-MISC", "O", "O", "O", "O", "O", "O", "B-PER", "O", "O"]},
  {"id": 32, "text": ["\"", "I", "thought", "it", "would", "be", "useful", "to", "have", "him", "back", "and", "he", "said", "he", "would", "be", "available", "."], "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
  {"id": 33, "text": ["I", "think", "now", "is", "the", "right", "time", "for", "him", "to", "return", ".", "\""], "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
  {"id": 34, "text": ["Squad", ":", "Javier", "Pertile", ",", "Paolo", "Vaccari", ",", "Marcello", "Cuttitta", ",", "Ivan", "Francescato", ",", "Leandro", "Manteri", ",", "Diego", "Dominguez", ",", "Francesco", "Mazzariol", ",", "Alessandro", "Troncon", ",", "Orazio", "Arancio", ",", "Andrea", "Sgorlon", ",", "Massimo", "Giovanelli", ",", "Carlo", "Checchinato", ",", "Walter", "Cristofoletto", ",", "Franco", "Properzi", "Curti", ",", "Carlo", "Orlandi", ",", "Massimo", "Cuttitta", ",", "Giambatista", "Croci", ",", "Gianluca", "Guidi", ",", "Nicola", "Mazzucato", ",", "Alessandro", "Moscardi", ",", "Andrea", "Castellani", "."], "labels": ["O", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O"]},
  {"id": 36, "text": ["SOCCER", "-", "LATE", "GOALS", "GIVE", "JAPAN", "WIN", "OVER", "SYRIA", "."], "labels": ["O", "O", "O", "O", "O", "B-LOC", "O", "O", "B-LOC", "O"]},
  {"id": 37, "text": ["AL-AIN", ",", "United", "Arab", "Emirates", "1996-12-06"], "labels": ["B-LOC", "O", "B-LOC", "I-LOC", "I-LOC", "O"]},
]
for example in examples:
  text = example["text"]
  text_input = " ".join(example["text"]) 
  print(text)
  labels = example["labels"]
  print(labels)
  tokens = tokenizer.tokenize(text_input)
  print(tokens)
  spans = tokenizer.encode_plus(text_input, return_offsets_mapping=True, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)["offset_mapping"]
  print(spans)

  # 先计算标签在真实文本里面的范围
  tmp_dict = {}
  for i, t in enumerate(text):
    if i == 0:
      start = 0
      end = len(t)
    else:
      start = end + 1
      end = start + len(t)
    tmp_dict[start] = ((start, end), labels[i])

  print(tmp_dict)

  # 这里重新计算标签
  new_labels = []
  pre_label = None
  for token, span in zip(tokens, spans):
    if span[0] in tmp_dict:
      new_labels.append(tmp_dict[span[0]][1])
      pre_end = span[1]
      pre_label = tmp_dict[span[0]][1].split("-")[-1]
    else:
      cur_label = "I-" + pre_label if pre_label != "O" else "O"
      new_labels.append(cur_label)

  print(new_labels)
  print("="*100)
         
"""
['Italy', 'recalled', 'Marcello', 'Cuttitta']
['B-LOC', 'O', 'B-PER', 'I-PER']
['Italy', 'recalled', 'Marcel', '##lo', 'Cut', '##ti', '##tta']
[(0, 5), (6, 14), (15, 21), (21, 23), (24, 27), (27, 29), (29, 32)]
{0: ((0, 5), 'B-LOC'), 6: ((6, 14), 'O'), 15: ((15, 23), 'B-PER'), 24: ((24, 32), 'I-PER')}
['B-LOC', 'O', 'B-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER']
====================================================================================================
['on', 'Friday', 'for', 'their', 'friendly', 'against', 'Scotland', 'at', 'Murrayfield', 'more', 'than', 'a', 'year', 'after', 'the', '30-year-old', 'wing', 'announced', 'he', 'was', 'retiring', 'following', 'differences', 'over', 'selection', '.']
['O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
['on', 'Friday', 'for', 'their', 'friendly', 'against', 'Scotland', 'at', 'Murray', '##field', 'more', 'than', 'a', 'year', 'after', 'the', '30', '-', 'year', '-', 'old', 'wing', 'announced', 'he', 'was', 'retiring', 'following', 'differences', 'over', 'selection', '.']
[(0, 2), (3, 9), (10, 13), (14, 19), (20, 28), (29, 36), (37, 45), (46, 48), (49, 55), (55, 60), (61, 65), (66, 70), (71, 72), (73, 77), (78, 83), (84, 87), (88, 90), (90, 91), (91, 95), (95, 96), (96, 99), (100, 104), (105, 114), (115, 117), (118, 121), (122, 130), (131, 140), (141, 152), (153, 157), (158, 167), (168, 169)]
{0: ((0, 2), 'O'), 3: ((3, 9), 'O'), 10: ((10, 13), 'O'), 14: ((14, 19), 'O'), 20: ((20, 28), 'O'), 29: ((29, 36), 'O'), 37: ((37, 45), 'B-LOC'), 46: ((46, 48), 'O'), 49: ((49, 60), 'B-LOC'), 61: ((61, 65), 'O'), 66: ((66, 70), 'O'), 71: ((71, 72), 'O'), 73: ((73, 77), 'O'), 78: ((78, 83), 'O'), 84: ((84, 87), 'O'), 88: ((88, 99), 'O'), 100: ((100, 104), 'O'), 105: ((105, 114), 'O'), 115: ((115, 117), 'O'), 118: ((118, 121), 'O'), 122: ((122, 130), 'O'), 131: ((131, 140), 'O'), 141: ((141, 152), 'O'), 153: ((153, 157), 'O'), 158: ((158, 167), 'O'), 168: ((168, 169), 'O')}
['O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
"""
