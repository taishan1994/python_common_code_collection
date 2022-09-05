import jieba

def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def get_word_input(sen, word2id, args):
    """这里并没有使用word2id，实际使用时要注意修改"""
    if len(sen) > args.max_seq_len - 2:
      sen = sen[:args.max_seq_len - 2]
    word_list = list(jieba.cut("".join(sen), cut_all=True))
    win = ['。' for i in range(6)]
    token_list = win + sen + win
    print(token_list)
    print(word_list)
    char_word = []
    # 遍历每一个token
    for i, token in enumerate(token_list[6:-6]):
        # 初始化[开始，中间，结尾，单个]
        char_word_id = [0, 0, 0, 0]
        # 如果该字符不是中文，直接跳过
        if not is_chinese(token):
            char_word.append(char_word_id)
            continue
        char_word_list = []
        # 指定范围内查找
        sub_text = ''.join(token_list[i:i+12])
        print(token, sub_text)
        # 如果在子串内发现词语，加入到char_word_list
        for word in word_list:
            if token in word and word in sub_text:
                char_word_list.append(word)

        for word in char_word_list:
            # word2id[word]
            try:
                if token == word:
                        # char_word_id[3] = word2id[word]
                        char_word_id[3] = word
                else:
                    index = word.index(token)
                    if index == 0:
                        # char_word_id[0] = word2id[word]
                        char_word_id[0] = word
                    elif index == len(word) - 1:
                        # char_word_id[2] = word2id[word]
                        char_word_id[2] = word
                    else:
                        # char_word_id[1] = word2id[word]
                        char_word_id[1] = word
            except Exception as e:
                continue
        char_word.append(char_word_id)
    # 加上CLS和SEP
    char_word = [[0,0,0,0]] + char_word + [[0,0,0,0]]
    while len(char_word) < args.max_seq_len:
        char_word.append([0 for i in range(4)])
    return char_word[:args.max_seq_len]


class Args:
    max_seq_len = 10

args = Args()
sen = "李明住在中山西路"
sen = [i for i in sen]
print(get_word_input(sen, word2id=None, args=args))
args = Args()

"""
Simplify the Usage of Lexicon in Chinese NER
如若按照论文所示的例子，下面的结果就不相同，而应该是根据词表构建前缀树，然后通过前缀树查找词语，而不是简单的通过结巴分词。
[[0, 0, 0, 0], [0, 0, 0, '李'], [0, 0, 0, '明'], [0, 0, 0, '住'], [0, 0, 0, '在'], ['中山西路', 0, 0, 0], ['山西路', '中山西路', '中山', 0], ['西路', '山西路', '山西', 0], [0, 0, '西路', 0], [0, 0, 0, 0]]
"""
