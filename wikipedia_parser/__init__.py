"""
This module contains code to work with the output of the wikipedia_parser Rust
utility.
"""

import os
from typing import Iterable, TextIO

import tqdm


class IndexedFlatFile:
    def __init__(
        self,
        index_filename: str,
        data_filename: str,
        *,
        show_progress: bool = False,
    ):
        """Open the index file, read it into memory and return an IndexedFlatFile."""
        index_file = open(index_filename)
        data_file = open(data_filename, "rb")
        file_len = os.fstat(data_file.fileno()).st_size
        if show_progress:
            index_file = tqdm.tqdm(index_file, desc="Reading index file")
        self.index_map = self.build_index(index_file, file_len)
        self.data_file = data_file

    @classmethod
    def build_index(
        cls, index_file: Iterable[str], file_len: int
    ) -> dict[str, tuple[int, int]]:
        index_map = {}
        last_key = None
        for line in index_file:
            offset, key = line.split(": ", maxsplit=1)
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
        """Get the JSON string containing a given article.
        The JSON structure is:
        article_schema = {
            "section_name": string (at the top level, this is the article name),
            "text": string (the main body of this section of the article),
            "children": [article_scema]
        }
        links_schema = {
            "title": string (article title),
            "links": [{
                "target": string (article canonical name used in URLs),
                "label": string (the text of the link)
            }]
        }
        """
        start, end = self.index_map[article_title]
        self.data_file.seek(start, os.SEEK_SET)
        bytes = self.data_file.read(end - start + 1)
        return bytes.decode("utf-8")
