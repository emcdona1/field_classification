import time


class Timer:
    def __init__(self, name: str):
        self.start_time: float = time.time()
        self.name: str = name
        self.duration: float = 0

    def start(self, name) -> None:
        if not self.start_time:
            self.start_time = time.time()
            print(f'Timer {self.name} started.')
        else:
            print(f'Timer {self.name} already started')

    def stop(self) -> None:
        end_time = time.time()
        self.duration = end_time - self.start_time
        print(f'Timer {self.name} stopped.')

    def print_results(self) -> None:
        print(f'Completed {self.name} in {self.duration:.1f} seconds.')

    def clear(self) -> None:
        self.start_time = None
        self.duration = None
