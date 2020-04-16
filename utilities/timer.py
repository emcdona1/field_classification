import time


class Timer:
    def __init__(self, name: str):
        self.start_time: float = time.time()
        self.name: str = name
        self.duration: float = 0

    def start(self, name) -> None:
        if not self.start_time:
            self.start_time = time.time()
            print('Timer %s started.' % self.name)
        else:
            print('Timer %s already started' % self.name)

    def stop(self) -> None:
        end_time = time.time()
        self.duration = end_time - self.start_time
        print('Timer %s stopped.' % self.name)

    def print_results(self) -> None:
        print('Completed %s in %.1f seconds.' % (self.name, self.duration))

    def clear(self) -> None:
        self.start_time = None
        self.duration = None
