from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import threading
import time


# 定义一个准备作为线程任务的函数
def task1():
    time.sleep(5)
    print(threading.current_thread().name)
    return 5


def task2():
    time.sleep(2)
    print(threading.current_thread().name)
    return 2

# ===========================================
# 实例1
# ===========================================
if __name__ == "__main__":
    # 创建一个包含2条线程的线程池
    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        # 向线程池提交一个task, pool.submit(任务名， 参数)
        future1 = pool.submit(task1)
        future2 = pool.submit(task2)
        all_task = [future1, future2]
        # 等待所有线程结束
        wait(all_task, return_when=ALL_COMPLETED)
        # 如果直接在这里获取了返回值，则上面那一句其实可以不需要了
        res1 = future1.result()
        res2 = future2.result()
        print(res1)
        print(res2)

    end = time.time()
    print('耗时：{}s'.format(end - start))

"""
结果：
<concurrent.futures.thread.ThreadPoolExecutor object at 0x7fdfbb050f28>_1
<concurrent.futures.thread.ThreadPoolExecutor object at 0x7fdfbb050f28>_0
5
2
耗时：5.0052244663238525s

"""
# ===========================================
# 实例2
# ===========================================
if __name__ == '__main__':
    # 创建一个包含2条线程的线程池
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=2)
    future1 = pool.submit(task1)
    future2 = pool.submit(task2)
    """
        当设置wait=True的时候会等待所有进程结束后再继续运行
            MainThread
            MainThread
            耗时：5.009037017822266s
        当设置wait=False的时候不会等待所有进程结束后再继续运行
            耗时：0.004068136215209961s
            MainThread
            MainThread
        不管设置未True还是设置为False，所有进程都会运行完毕
    """
    pool.shutdown(wait=False)
    end = time.time()
    print('耗时：{}s'.format(end - start))
    

# ===========================================
# 实例3
# ===========================================
if __name__ == '__main__':
    # 创建一个包含2条线程的线程池
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=2)
    future1 = pool.submit(task1)
    future2 = pool.submit(task2)
    print(future1.result())
    print(future2.result())
    """
        当设置wait=True的时候会等待所有进程结束后再继续运行
            MainThread
            MainThread
            耗时：5.009037017822266s
        当设置wait=False的时候不会等待所有进程结束后再继续运行
            耗时：0.004068136215209961s
            MainThread
            MainThread
        不管设置未True还是设置为False，所有进程都会运行完毕
    """
    pool.shutdown(wait=False)  # 由于上面使用了.result()，因此这里无论是设置为True还是False都会先等任务线程完成
    end = time.time()
    print('耗时：{}s'.format(end - start))
"""
<concurrent.futures.thread.ThreadPoolExecutor object at 0x7fc56e050f60>_1
<concurrent.futures.thread.ThreadPoolExecutor object at 0x7fc56e050f60>_0
5
2
耗时：5.005308628082275s
"""

# ===========================================
# 实例4，主要是map函数的使用
# ===========================================
if __name__ == '__main__':
    # 创建一个包含2条线程的线程池
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=4)
    """
        也可以使用map函数，map第一个参数为函数，第二个参数为函数所需参数
        由于：
            for i in iters:
                print(i)
        因此会先等所有线程完成再继续，否则会先运行主程序
    """
    iters = pool.map(task, [random.randint(1, 3) for _ in range(3)])
    for i in iters:
        print(i)
    end = time.time()
    print('耗时：{}s'.format(end - start))
