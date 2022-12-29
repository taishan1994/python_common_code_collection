import jieba

jieba.add_word("[CLS]")
jieba.add_word("[SEP]")

texts = [
    "武汉市长江大桥"
]


def build_word_vocab(texts):
    """可用于将词转换为id"""
    word_vocabs = []
    for text in texts:
        words = jieba.lcut(text, cut_all=True)
        word_vocabs.extend(words)
    word_vocabs = list(set(word_vocabs))
    return word_vocabs


def char_with_words(sen):
    words = jieba.lcut(sen, cut_all=True)
    tokens = ['[CLS]'] + [i for i in sen] + ['[SEP]']
    extra = ['#' for i in range(6)]
    tokens = extra + tokens + extra
    char_words = []
    for i, token in enumerate(tokens[6:-6]):
        char_word = []
        sub_text = "".join(tokens[i:i + 12])
        for word in words:
            if token in word and word in sub_text:
                char_word.append(word)
        # 这里直观点没转换为id
        char_word_ids = ["null", "null", "null", "null"]
        for word in char_word:
            if token == word:
                char_word_ids[3] = word
            else:
                ind = word.index(token)
                if ind == 0:
                    char_word_ids[0] = word
                elif ind == len(word) - 1:
                    char_word_ids[2] = word
                else:
                    char_word_ids[1] = word

        char_words.append(char_word_ids)
    for char, c_word in zip(tokens[6:-6], char_words):
        print(char, c_word)


char_with_words("武汉市长江大桥")
