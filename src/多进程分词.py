import sys

sys.path.append('..')
import jieba
from keyword_extraction.utils import get_stopwords, get_all_docs

"""
多进程分词，参考：https://blog.csdn.net/qq_43520571/article/details/116493735
"""

stopwords = get_stopwords()


def segment(line, cut_all=True):
    try:
        gen = jieba.lcut(line, cut_all=cut_all)  # 对字符串前一百分词
    except Exception as e:
        print(e)
        return ""
    if stopwords:
        words = [i for i in gen if (i not in stopwords
                                    and i != ''
                                    and i != '\n')]
    else:
        words = " ".join(gen)
    return ' '.join(words)  # 空格分割的字符串


def run_imap_mp(func, argument_list, num_processes=None, is_tqdm=True):
    '''
    多进程与进度条结合
    这里使用的imap返回的结果是有序的
    param:
    ------
    func:function
        函数
    argument_list:list
        参数列表
    num_processes:int
        进程数，不填默认为总核心
    is_tqdm:bool
        是否展示进度条，默认展示
    '''
    result_list_tqdm = []
    try:
        import multiprocessing
        if not num_processes:
            num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)
        if is_tqdm:
            from tqdm import tqdm
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
        else:
            for result in pool.imap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
    except:
        result_list_tqdm = list(map(func, argument_list))
    return result_list_tqdm


def main():
    docs = get_all_docs()
    res = run_imap_mp(segment, docs, num_processes=2)
    print(res[0])
    with open('../data/segment.txt', 'w', encoding='utf-8') as fp:
        fp.write("\n".join(res))


if __name__ == '__main__':
    main()
