def cut_sentences_main(text, max_seq_len):
    if len(text) <= max_seq_len:
        return [text]
    try:
        text = re.split(r"([。!！?？])", text)  # 先按照粗粒度近进行切句
        text.append("")
        text = ["".join(i) for i in zip(text[0::2], text[1::2])]
        # print(len("".join(text)))
        res = []
        res_tmp = []
        for tex in text:
            res_tmp.append(tex)
            if len("".join(res_tmp)) > max_seq_len:  # res_tmp里面句子太长了
                res.append("".join(res_tmp[:-1]))  # 不要最后的那一句
                res_tmp = [res_tmp[-1]]
                if len("".join(res_tmp)) > max_seq_len:  # 最后一句也太长了，则进行细粒度切分
                    tex2 = re.split(r"([，,])", tex)
                    tex2.append("")
                    tex2 = ["".join(i) for i in zip(text[0::2], text[1::2])]
                    res_tmp2 = []
                    for i, te2 in enumerate(tex2):
                        res_tmp2.append(te2)
                        if len("".join(res_tmp2)) > max_seq_len:
                            res.append("".join(res_tmp2[:-1]))
                            res_tmp = tex2[i:]
                            break
        if res_tmp:
            res.append("".join(res_tmp))
        return res
    except Exception as e:
        res = []
        batch = len(text) // max_seq_len
        for i in range(batch + 1):
            res.append(text[i * max_seq_len:(i + 1) * max_seq_len])
        return res
