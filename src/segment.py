# coding=utf-8
import jieba
import thulac
import pkuseg
import stanza
import pandas as pd
from ltp import LTP
from pyhanlp import *

"""
pip install jieba
pip install thulac==0.2.2
pip install pkuseg
pip install ltp
pip install stanza
"""


def get_data():
    data = pd.read_excel("./data/result_2023022315.xlsx", encoding="gbk")
    jyaq = data["简要案情"].tolist()
    return jyaq


def seq_by_jieba(d):
    return jieba.lcut(d, cut_all=False)


def seg_by_thulac(d, thul):
    """
    thulac(user_dict=None, model_path=None, T2S=False, seg_only=False, filt=False)
    初始化程序，进行自定义设置
    user_dict：设置用户词典，用户词典中的词会被打上uw标签。词典中每一个词一行，UTF8编码
    T2S：默认False, 是否将句子从繁体转化为简体
    seg_only：默认False, 时候只进行分词，不进行词性标注
    filt：默认False, 是否使用过滤器去除一些没有意义的词语，例如“可以”。
    model_path：设置模型文件所在文件夹，默认为models /
    """
    return thul.cut(d, text=False)


def seg_by_pkuseg(d, seg):
    return seg.cut(d)


def seg_by_ltp(d, ltp):
    return ltp.seg([d])[0][0]


def seg_by_hanlp(d, analyzer):
    return analyzer.analyze(d)


def seg_by_stanza(d, zh_nlp):
    tmp = zh_nlp(d)
    for sen in tmp.sentences:
        return [token.text for token in sen.tokens]



def main():
    data = get_data()
    thul = thulac.thulac(seg_only=True)
    pkus = pkuseg.pkuseg()
    ltp = LTP()
    CRFLexicalAnalyzer = JClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")
    analyzer = CRFLexicalAnalyzer()
    PerceptronLexicalAnalyzer = JClass("com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer")
    analyzer = PerceptronLexicalAnalyzer()
    HMMLexicalAnalyzer = JClass("com.hankcs.hanlp.model.hmm.HMMLexicalAnalyzer")
    analyzer = PerceptronLexicalAnalyzer()
    zh_nlp = stanza.Pipeline('zh', processors="tokenize", dir="models/default/", model_dir="models/default/", )
    for i, d in enumerate(data):
        if i > 10:
            break
        d = d.strip()
        # d = seq_by_jieba(d)
        # d = seg_by_thulac(d, thul)
        # d = seg_by_pkuseg(d, pkus)
        # d = seg_by_ltp(d, ltp)
        # d = seg_by_hanlp(d, analyzer)
        d = seg_by_stanza(d, zh_nlp)
        print(d)


if __name__ == '__main__':
    main()

