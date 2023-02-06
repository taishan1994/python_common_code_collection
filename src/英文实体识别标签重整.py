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
