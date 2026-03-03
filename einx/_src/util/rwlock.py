import threading
import contextlib


class RWLock:
    def __init__(self):
        self.write_lock = threading.RLock()
        self.num_reading_lock = threading.RLock()
        self.num_reading = 0

    def read_acquire(self):
        with self.num_reading_lock:
            self.num_reading += 1
            if self.num_reading == 1:  # First reader acquires write lock
                self.write_lock.acquire()

    def read_release(self):
        assert self.num_reading > 0
        with self.num_reading_lock:
            self.num_reading -= 1
            if self.num_reading == 0:  # Last reader releases write lock
                self.write_lock.release()

    @contextlib.contextmanager
    def read(self):
        try:
            self.read_acquire()
            yield
        finally:
            self.read_release()

    def write_acquire(self):
        self.write_lock.acquire()

    def write_release(self):
        self.write_lock.release()

    @contextlib.contextmanager
    def write(self):
        try:
            self.write_acquire()
            yield
        finally:
            self.write_release()
