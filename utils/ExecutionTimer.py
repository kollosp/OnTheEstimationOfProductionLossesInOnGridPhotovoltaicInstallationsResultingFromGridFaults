import time

class ExecutionTimer:
    def __init__(self):
        self.start = 0
        self.end = 0

    @property
    def seconds_elapsed(self):
        return self.end - self.start

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        # Exception handling here
        self.end = time.time()