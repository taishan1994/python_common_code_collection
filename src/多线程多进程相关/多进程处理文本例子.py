from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, ProcessPoolExecutor
import threading
import time
import random


# 定义一个准备作为线程任务的函数
def task(arg):
    num = arg[0]
    ind = arg[1]
    text = arg[2]
    time.sleep(num)
    print(threading.current_thread().name, num)
    token = [i for i in text]
    return (ind, token)


if __name__ == '__main__':
    sleep_time = [2,3,1]
    texts = [
        '中国是一个和平、自由的国家',
        '武汉市长江大桥',
        '我也想过过过儿过过的生活',
    ]
    args = []
    for i in range(3):
        args.append([sleep_time[i], i, texts[i]])
    start = time.time()
    pool = ProcessPoolExecutor(max_workers=4)
    iters = pool.map(task, args)
    res = []
    for i in iters:
        print(i)
        res.append(i)
    print(res)
    end = time.time()
    print('耗时：{}s'.format(end - start))

    
"""
MainThread 1
MainThread 2
(0, ['中', '国', '是', '一', '个', '和', '平', '、', '自', '由', '的', '国', '家'])
MainThread 3
(1, ['武', '汉', '市', '长', '江', '大', '桥'])
(2, ['我', '也', '想', '过', '过', '过', '儿', '过', '过', '的', '生', '活'])
[(0, ['中', '国', '是', '一', '个', '和', '平', '、', '自', '由', '的', '国', '家']), (1, ['武', '汉', '市', '长', '江', '大', '桥']), (2, ['我', '也', '想', '过', '过', '过', '儿', '过', '过', '的', '生', '活'])]
耗时：3.008471727371216s
"""
