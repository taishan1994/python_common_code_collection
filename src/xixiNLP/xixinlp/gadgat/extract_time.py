import time


class TimeParser:
    def __init__(self):
        self.cur_time = time.time()

    def __call__(self, cur_time, *args, **kwargs):
        self.cur_time = cur_time
        return self.cur_time
