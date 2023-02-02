from transformers import BertTokenizer, BertModel
eng_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

text1 = ['West', 'Indian', 'all-rounder', 'Phil', 'Simmons', 'took', 'four', 'for', '38', 'on', 'Friday', 'as', 'Leicestershire', 'beat', 'Somerset', 'by', 'an', 'innings', 'and', '39', 'runs', 'in', 'two', 'days', 'to', 'take', 'over', 'at', 'the', 'head', 'of', 'the', 'county', 'championship', '.']
example = {"id": 64, "text": ["June", "25-27", "v", "British", "Universities", "(", "at", "Oxford", ",", "three", "days", ")"], "labels": ["O", "O", "O", "B-ORG", "I-ORG", "O", "O", "B-LOC", "O", "O", "O", "O"]}
example = {"id": 351, "text": ["the", "$", "1.2", "million", "Greater", "Milwaukee", "Open", "at", "the", "par-71", ","], "labels": ["O", "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "O", "O", "O"]}
label = example["labels"]
text1 = example["text"]
text2 = tokenizer.tokenize(" ".join(text1))
print(text1)
print(text2)
print(label)
def align_label_example(input, tokenized_input, label, label2id, label_all_tokens=False):
  i, j = 0, 0
  ids = []
  # 这里需要遍历tokenizer后的列表
  while i < len(tokenized_input):
    if tokenized_input[i] == input[j]:
      ids.append(label2id[label[j]])
      # ids.append(label[j])
      i += 1
      j += 1
    else:
      tmp_word = tokenized_input[i]
      tmp = []
      while i < len(tokenized_input) and "".join(tmp) != input[j]:
        tmp_word = tokenized_input[i][2:] if tokenized_input[i][:2] == "##" else tokenized_input[i]
        tmp.append(tmp_word)
        if label[j] == "O":
          ids.append(label2id[label[j]])
        else:
          ids.append(label2id[label[j]] if label_all_tokens else -100)
        # ids.append(label[j])
        i += 1
      j += 1
  assert len(ids) == len(tokenized_input)
  return ids

def conver_labels_to_biolabels(labels):
  label2id = {"O":0}
  id2label = {0:"O"}
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

with open("data/conll2003/mid_data/labels.txt", "r") as fp:
  labels = fp.read().strip().split("\n")
labels = ['PER', 'LOC', 'MISC', 'ORG']
label2id, id2label = conver_labels_to_biolabels(labels)
align_label_example(text1, text2, label, label2id, label_all_tokens=True)

"""
['June', '25-27', 'v', 'British', 'Universities', '(', 'at', 'Oxford', ',', 'three', 'days', ')']
['June', '25', '-', '27', 'v', 'British', 'Universities', '(', 'at', 'Oxford', ',', 'three', 'days', ')']
['O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O']
[0, 0, 0, 0, 0, 7, 8, 0, 0, 3, 0, 0, 0, 0]
"""
