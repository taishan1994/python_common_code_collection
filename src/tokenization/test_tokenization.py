"""
该文件用于将tokenization化后的文本转换为命名实体识别相关标签
"""
from tokenization import BasicTokenizer
from transformers import BertTokenizer

basicTokenizer = BasicTokenizer(do_lower_case=False)
bertTokenizer = BertTokenizer.from_pretrained('./')
text = '我喜欢玩computer games，你喜欢play football吗？'
locs = [('computer games', 'entity', 4, 17), ('play football', 'entity', 22, 34)]
print(text[4:18], text[22:35])

process_text = []
process_char = []
process_labels = []
for i,loc in enumerate(locs):
    [entity, type, start, end] = loc
    if i == 0:
        process_text.append(text[:start])
        process_text.append(text[start:end+1])
        process_labels += ['O'] * len(text[:start])
        tmp_text = basicTokenizer.tokenize(entity)
        process_char += text[:start]
        process_char += tmp_text
        tmp_labels = [0] * len(tmp_text)
        for j in range(len(tmp_text)):
            if j == 0:
                tmp_labels[j] = 'B-' + type
            else:
                tmp_labels[j] = 'I-' + type
    else:
        [tmp_entity, tmp_type, tmp_start, tmp_end] = locs[i-1]
        process_text.append(text[tmp_end+1:start])
        process_text.append(text[start:end+1])
        process_labels += ['O'] * len(text[tmp_end+1:start])
        tmp_text = basicTokenizer.tokenize(entity)
        process_char += text[tmp_end+1:start]
        process_char += tmp_text
        tmp_labels = [0] * len(tmp_text)
        for j in range(len(tmp_text)):
            if j == 0:
                tmp_labels[j] = 'B-' + type
            else:
                tmp_labels[j] = 'I-' + type
    process_labels += tmp_labels
    if i == len(locs) - 1:
        process_text.append(text[end+1:])
        process_char += text[end+1:]
        process_labels += ['O'] * len(text[end+1:])

print(process_text)
print(process_labels)
print(process_char)

# tokens = basicTokenizer.tokenize(text)
# print(tokenss)

"""
computer games play football
['我喜欢玩', 'computer games', '，你喜欢', 'play football', '吗？']
['O', 'O', 'O', 'O', 'B-entity', 'I-entity', 'O', 'O', 'O', 'O', 'B-entity', 'I-entity', 'O', 'O']
['我', '喜', '欢', '玩', 'computer', 'games', '，', '你', '喜', '欢', 'play', 'football', '吗', '？']
"""
