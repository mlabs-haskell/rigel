"""
This module contains code to work with the output of the wikipedia_parser Rust
utility.
"""

import os
from typing import Iterable, TextIO

import tqdm


def file_len(file) -> int:
    return os.fstat(file.fileno()).st_size


class IndexedFlatFile:
    def __init__(
        self,
        index_filename: str,
        data_filename: str,
        *,
        show_progress: bool = False,
    ):
        """Open the index file, read it into memory and return an IndexedFlatFile."""

        self.index_filename = index_filename
        self.data_filename = data_filename

        self.data_file = open(data_filename, "rb")

        self.build_index(show_progress=show_progress)

    def build_index(self, *, show_progress: bool = False):
        index_file = open(self.index_filename)
        index_file_len = file_len(index_file)

        progress_bar = None
        if show_progress:
            print("Reading index file:")
            progress_bar = tqdm.tqdm(
                desc=self.index_filename,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=index_file_len,
            )

        data_file_len = file_len(self.data_file)

        index_map = {}
        last_key = None
        while line := index_file.readline():
            offset, key = line.split(": ", maxsplit=1)
            key = key.rstrip("\n")
            offset = int(offset)
            if last_key is not None:
                start, end = index_map[last_key]
                end = offset - 1
                index_map[last_key] = (start, end)

            start = offset
            end = data_file_len
            index_map[key] = (start, end)
            last_key = key

            if progress_bar is not None:
                progress_bar.update(index_file.tell() - progress_bar.n)

        self.index_map = index_map

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
