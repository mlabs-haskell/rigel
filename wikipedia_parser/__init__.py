"""
This module contains code to work with the output of the wikipedia_parser Rust
utility.
"""

import os
from typing import Iterable, TextIO

Index = dict[str, tuple[int, int]]


class IndexedFlatFile:
    def __init__(self, index: Index, data_file: TextIO):
        self.index: Index = index
        self.data_file = data_file

    @classmethod
    def open(cls, index_file: Iterable[str], data_file: TextIO):
        """Open the index file, read it into memory and return an IndexedFlatFile."""
        file_len = os.fstat(data_file.fileno()).st_size
        index = cls.build_index(index_file, file_len)
        return cls(index, data_file)

    @classmethod
    def build_index(cls, index_file: Iterable[str], file_len: int):
        index = {}
        last_key = None
        for line in index_file:
            offset, key = line.split(":", maxsplit=1)
            key = key.rstrip("\n")
            offset = int(offset)
            if last_key is not None:
                start, end = index[last_key]
                end = offset - 1
                index[last_key] = (start, end)

            start = offset
            end = file_len
            index[key] = (start, end)
            last_key = key
        return index

    def get(self, key: str) -> str:
        start, end = self.index[key]
        self.data_file.seek(start)
        return self.data_file.read(end - start + 1)
