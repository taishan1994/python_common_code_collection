import time


def print_run_time(func):
    """时间装饰器"""

    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print('[%s] run time is %.4f' % (func.__name__, time.time() - local_time))
        return res

    return wrapper
