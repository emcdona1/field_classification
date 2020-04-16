import time


class Timer:
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name
        self.duration = None

    def stop(self):
        end_time = time.time()
        self.duration = end_time - self.start_time

    def print_results(self):
        print('Completed %s in %.1f seconds' % (self.name, self.duration))

    def clear(self):
        self.start_time = None
        self.duration = None

    def start(self, name):
        if not self.start_time:
            self.start_time = time.time()
            self.name = name
        else:
            print('Timer %s already started' % self.name)
