from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, ProcessPoolExecutor
from loguru import logger
import time
import threading

"""
   默认情况下是线程安全的
   logger.add()可选参数：
       rotation="12:00"  # 每天12点创建一个新的文件
       rotation="1 MB"  # 文件超过1MB就创建一个新的文件
       compression="zip"  # 压缩日志
       retention=“10 days”  # 最长保留时间
   如果想多进程记录日志：enqueue=True
       backtrace=True  # 打印整个堆栈信息，使用时：
       try:
           func()
       excep Exception as e:
           logger.exception("自定义标识")
   邮件告警：可以和强大的邮件通知模块 notifiers 库结合使用
   可以作为装饰器使用：@logger.catch
   :return:
"""

logger.add("test.log", enqueue=True)

def task1():
    time.sleep(5)
    logger.info(threading.current_thread().name)
    return 5


def task2():
    time.sleep(3)
    logger.info(threading.current_thread().name)
    return 3


if __name__ == "__main__":
    # 创建一个包含2条线程的线程池
    start = time.time()
    with ProcessPoolExecutor(max_workers=2) as pool:
        # 向线程池提交一个task, pool.submit(任务名， 参数)
        future1 = pool.submit(task1)
        future2 = pool.submit(task2)
        all_task = [future1, future2]
        # 等待所有线程结束
        wait(all_task, return_when=ALL_COMPLETED)
        # 如果直接在这里获取了返回值，则上面那一句其实可以不需要了
        res1 = future1.result()
        res2 = future2.result()
        logger.info(res1)
        logger.info(res2)

    end = time.time()
    logger.info('耗时：{}s'.format(end - start))

"""
2022-11-16 16:21:52.823 | INFO     | __mp_main__:task2:34 - MainThread
2022-11-16 16:21:54.824 | INFO     | __mp_main__:task1:28 - MainThread
2022-11-16 16:21:54.824 | INFO     | __main__:<module>:51 - 5
2022-11-16 16:21:54.824 | INFO     | __main__:<module>:52 - 3
2022-11-16 16:21:54.864 | INFO     | __main__:<module>:55 - 耗时：5.360519647598267s
"""
