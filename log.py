# coding: utf-8
# pylint: disable=missing-docstring, invalid-name


from multiprocessing import Lock

class Log:
    def __init__(self):
        self.lock = Lock() # use between process to prevent intermingled logs

    def log(self, txt):
        # prevent intermingled logs
        with self.lock:
            print (f'> {txt}')

log = Log().log
