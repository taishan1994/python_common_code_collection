import time


def print_run_time(func):
    """时间装饰器"""

    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print('[%s] run time is %.4f' % (func.__name__, time.time() - local_time))

    return wrapper


@print_run_time
def test():
    time.sleep(5)
    print('test运行完毕')


if __name__ == '__main__':
    test()

"""
test运行完毕
[test] run time is 5.0052
"""
