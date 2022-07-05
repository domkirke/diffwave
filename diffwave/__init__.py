
from typing import Iterator

class IterableConfig(Iterator):
    def __init__(self, config):
        self.config = config

    def __next__(self):
        while True:
            return self.config
