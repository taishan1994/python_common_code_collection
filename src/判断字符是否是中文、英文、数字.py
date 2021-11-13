"""
补充：python自带的有isdigit()、isalpha()、isalnum()三个函数
    1.python官方定义中的字母：大家默认为英文字母+汉字即可
    2.python官方定义中的数字：大家默认为阿拉伯数字+带圈的数字即可
    
    S.isdigit()返回的是布尔值：True False
    S中至少有一个字符且如果S中的所有字符都是数字，那么返回结果就是True；否则，就返回False
    
    S.isalpha()返回的是布尔值：True False
    S中至少有一个字符且如果S中的所有字符都是字母，那么返回结果就是True；否则，就返回False
    
    S.isalnum()返回的是布尔值：True False
    S中至少有一个字符且如果S中的所有字符都是字母数字，那么返回结果就是True；否则，就返回False
"""

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


if __name__ == '__main__':
    str = "我爱北京天安门!xiximayou_333"
    for i in str:
        print('=================================')
        print(i, '是字母' if is_alphabet(i) else '不是字母')
        print(i, '是中文' if is_chinese(i) else '不是中文')
        print(i, '是数字' if is_number(i) else '不是数字')
        print(i, '不是中文、数字、英文' if is_other(i) else '是中文或者数字或者英文')
        print('=================================')
