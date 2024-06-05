import gzip
from typing import Iterator, TypedDict
import bson
import config


class Doc(TypedDict):
    _id: bson.ObjectId
    title: str
    context_vector: list[list[float]]
    header_name: str


def open_dump_file(filename: str) -> Iterator[Doc]:
    file = gzip.open(filename, "rb")
    iter = bson.decode_file_iter(file)  # type: ignore
    return iter  # type: ignore


def open_dump_file_by_index(i: int) -> Iterator[Doc]:
    return open_dump_file(config.DUMPS_DIR + f"/bin_{i}.bson.gz")
