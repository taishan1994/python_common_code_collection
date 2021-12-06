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


# 创建一个包含2条线程的线程池
start = time.time()
with ThreadPoolExecutor(max_workers=2) as pool:
    # 向线程池提交一个task, pool.submit(任务名， 参数)
    future1 = pool.submit(task1)
    future2 = pool.submit(task2)
    all_task = [future1, future2]
    # 等待所有线程结束
    wait(all_task, return_when=ALL_COMPLETED)
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

