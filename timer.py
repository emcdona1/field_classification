import time


class Timer:
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def stop(self):
        end_time = time.time()
        print('Completed %s in %.1f seconds' % (self.name, end_time - self.start_time))
        self.start_time = None

    def start(self, name):
        if not self.start_time:
            self.start_time = time.time()
            self.name = name
        else:
            print('Timer %s already started' % self.name)
