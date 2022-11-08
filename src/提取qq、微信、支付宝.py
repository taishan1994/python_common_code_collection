import re
import pandas as pd


end_filter_list = ['钱', '元', '块', '月', '年', '月', '日', '多']


def filter_zhifubao(zhifubao, start, end, content):
    if len(zhifubao) < 5:
        return False
    if end < len(content) and content[end] in end_filter_list:
        return False
    if zhifubao in ["email", "Email"]:
        return False
    return True

def filter_qq(qq, start, end, content):
    if len(qq) < 5:
        return False
    if end < len(content) and content[end] in end_filter_list:
        return False
    if qq in ["email", "Email"]:
        return False
    if end < len(content) and end + 1 < len(content) and content[end] == "." \
            and content[end + 1] in [str(i) for i in range(10)]:
        return False
    return True


def filter_weixin(weixin, start, end, content):
    if len(weixin) < 5:
        return False
    if end < len(content) and content[end] in end_filter_list:
        return False
    if weixin in ["email", "Email"]:
        return False
    if end < len(content) and end + 1 < len(content) and content[end] == "." \
            and content[end + 1] in [str(i) for i in range(10)]:
        return False
    return True


def extract_one_main(bl):
    try:
        qq_res, weixin_res = extract_qq_and_weixin(bl)
        zhifubao_res = extract_zhifubao(bl)
        if qq_res or weixin_res or zhifubao_res:
            print(zhifubao_res)
            print(qq_res, weixin_res)
            print("=" * 100)
    except Exception as e:
        qq_res = []
        weixin_res = []
        zhifubao_res = []
        print(e)


def extract_zhifubao(bl):
    try:
        zhifubao_res = []
        for content in bl:
            if re.search("支付宝", content):
                span = re.finditer("[0-9a-zA-Z@\.@]+", content)
                for s in span:
                    start = s.start()
                    end = s.end()
                    text = content[start:end]
                    before_text = content[0 if start - 10 < 0 else start - 10:start]
                    if "支付宝" not in before_text or \
                            not filter_zhifubao(text, start, end, content) \
                            or re.search("(订单|密码|昵称)", content):
                        continue
                    if text[-1] == ".":
                        text = text[:-1]
                    # print((text, content))
                    zhifubao_res.append(text)
    except Exception as e:
        # print(e)
        pass

    return list(set(zhifubao_res))


def extract_qq_and_weixin(bl):
    try:
        qq_res = []
        weixin_res = []
        qq_end_ind = []
        weixin_end_ind = []
        for content in bl:
            if re.search("(qq|QQ|微信)", content):  # 判断文本中是否包含关键字
                span = re.finditer("[0-9a-zA-Z_-]+", content)
                if span:
                    for s in span:
                        start = s.start()
                        end = s.end()
                        text = content[start:end]
                        before_text = content[0 if start - 10 < 0 else start - 10:start]
                        if re.search("(qq|QQ|Qq|qq)", before_text) or \
                                (end < len(content) and content[end:end + 3] == "@qq"):
                            if filter_qq(text, start, end, content):
                                qq_end_ind.append(end)
                                # qq_res.append((text, content))
                                qq_res.append(text)
                        elif re.search("微信", before_text) and \
                                not re.search("(微信名|昵称|号名|手机)", before_text):
                            if filter_weixin(text, start, end, content):
                                weixin_end_ind.append(end)
                                # weixin_res.append((text, content))
                                weixin_res.append(text)
                        else:
                            # 这里是避免两个号码连在一起没有识别出来
                            if start - 1 >= 0 and content[start - 1] in ['，', ',', '、']:
                                if start - 1 in qq_end_ind:
                                    # qq_res.append((text, content))
                                    qq_res.append(text)
                                elif start - 1 in weixin_end_ind:
                                    # weixin_res.append((text, content))
                                    weixin_res.append(text)
        # if qq_res or weixin_res:
        #     for i in qq_res:
        #         print(i)
        #     for i in weixin_res:
        #         print(i)
    except Exception as e:
        print(e)
        pass
    return list(set(qq_res)), list(set(weixin_res))


if __name__ == '__main__':
    bl = "问:这个客户的联系方式?\n答:微信号:123123-888,weawew."
    bl = "QQ号码是:1234124、12421421"
    extract_one_main(bl)
