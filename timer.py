import time


class Timer:
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def stop_timer(self):
        end_time = time.time()
        print('Completed in %.1f seconds' % (end_time - self.start_time))

    def start_time(self):
        self.start_time = time.time()
