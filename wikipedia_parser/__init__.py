"""
This module contains code to work with the output of the wikipedia_parser Rust
utility.
"""

import os
from typing import Iterable, TextIO

class IndexedFlatFile:
    def __init__(self, index_file: Iterable[str], data_file: TextIO):
        """Open the index file, read it into memory and return an IndexedFlatFile."""
        file_len = os.fstat(data_file.fileno()).st_size
        self.index_map = self.build_index(index_file, file_len)
        self.data_file = data_file

    @classmethod
    def build_index(
        cls,
        index_file: Iterable[str],
        file_len: int
    ) -> dict[str, tuple[int, int]]:
        index_map = {}
        last_key = None
        for line in index_file:
            offset, key = line.split(":", maxsplit=1)
            key = key.rstrip("\n")
            offset = int(offset)
            if last_key is not None:
                start, end = index_map[last_key]
                end = offset - 1
                index_map[last_key] = (start, end)

            start = offset
            end = file_len
            index_map[key] = (start, end)
            last_key = key
        return index_map

    def get(self, article_title: str) -> str:
        """Get the JSON string containing a given article. The JSON structure is:
        article_schema = {
            "section_name": string (at the top level, this is the article name),
            "text": string (the main body of this section of the article),
            "children": [article_scema]
        }
        """
        start, end = self.index_map[article_title]
        self.data_file.seek(start)
        return self.data_file.read(end - start + 1)
