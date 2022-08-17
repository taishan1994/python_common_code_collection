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
    if len(text) <= max_seq_len:
        return [text]

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


if __name__ == '__main__':
    text = "2021年12月31日18时11分许，事主陈明刚（13713897598）亲临我所报警称：于2021年12月30日17时10分许，其将电动车停放在广东省深圳市宝安区建安新村3栋2单元楼下，于2021年12月31日13时许，其用电动车时发现电动车后轮胎发现没有气了，发现其电动车后车胎有一个大概2cm的口子，遂报警。"
    text = "我所跟进情报在龙岗区坪地街道青顶背7号206抓获一对卖淫嫖娼人员。张青荣（430425197802255074）、欧晓燕（522631199111110440），"
    text = "其向对方提供的工商银行(卡号:6212261714011793739,户主:谢波)、恒丰银行(卡号:6230780100012591260,户主:何庆伟)、交通银行(卡号:6222621310018572574,户主:王明和)、兴业银行,(卡号:622908383033407408,户主:郭超)、建设银行(卡号:6217004260030205165,户主:赵晓瑞)、广东农村信用社联合社(卡号:6217281542901737366,户主:王建业)、(卡号:6217281542901737424,户主:杨再松)、长沙银行(卡号:6214467873167269973,户主:胡卓)、湖南省农村信用社(卡号:6230901818138729050,户主:谢波)、河南省农村信用联合社(卡号:623059100602098581,户主:徐文彬)、中国邮政储蓄银行(卡号:6221805530000609825,户主:谢昆程)、威海市商业银行(卡号:6231020101009337319,户主:郝华伟)、平安银行(卡号:6221551883761605,户主:李敬杰)共计转账483047元。"
    res = cut_sentences_main(text, 512)
    print(res)
    for i in res:
        print(len(i))
        print(i)
