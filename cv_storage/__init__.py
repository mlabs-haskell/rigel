from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
import numpy as np
import struct


@dataclass
class CVMetadata:
    start: int
    end: int  # Note: We don't _need_ to store end, but it simplifies the read logic.
    article_title: str
    section_name: str


class CVMetadataCache:
    def __init__(self):
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def add(self, metadata: CVMetadata):
        title = metadata.article_title
        section = metadata.section_name
        if title not in self.cache:
            self.cache[title] = {}
        self.cache[title][section] = metadata

    def get(self, article_title: str, section_name: str) -> CVMetadata | None:
        if article_title not in self.cache:
            return None
        return self.cache[article_title].get(section_name)

    def has_article(self, article_title: str) -> bool:
        return article_title in self.cache

    def get_article_titles(self) -> list[str]:
        return list(self.cache.keys())

    def get_section_names(self, article_title: str) -> list[str]:
        if article_title not in self.cache:
            return []
        return list(self.cache[article_title].keys())


class ContextVectorDB:
    """Class for saving and loading context vectors"""

    def __init__(self, folder: Path):
        if folder.exists():
            if not folder.is_dir():
                raise ValueError("Given path is not a folder: " + str(folder))
        else:
            folder.mkdir()
        index_file_path = folder / "index.cvdb"
        data_file_path = folder / "data.cvdb"

        self.metadata_file = open(index_file_path, "a+b")
        self.data_file = open(data_file_path, "a+b")
        self.metadata_cache = CVMetadataCache()
        self._build_metadata_cache()

    def __del__(self):
        self.metadata_file.close()
        self.data_file.close()

    # Public interface

    def get(self, article_title: str, section_name: str) -> np.ndarray | None:
        metadata = self.metadata_cache.get(article_title, section_name)
        if metadata is None:
            return None
        return self._read_context_vector(metadata.start)

    def has_article(self, article_title: str) -> bool:
        return self.metadata_cache.has_article(article_title)

    def has_section(self, article_title: str, section_name: str) -> bool:
        metadata = self.metadata_cache.get(article_title, section_name)
        return metadata is not None

    def get_article_titles(self) -> list[str]:
        return self.metadata_cache.get_article_titles()

    def get_section_names(self, article_title: str) -> list[str]:
        return self.metadata_cache.get_section_names(article_title)

    def insert(self, article_title: str, section_name: str, cv: np.ndarray):
        start = self.data_file.tell()
        self._write_context_vector(cv)
        end = self.data_file.tell()

        metadata = CVMetadata(start, end, article_title, section_name)
        self._write_metadata(metadata)
        self._cache_metadata(metadata)

    # Context vectors

    def _read_context_vector(self, start):
        self.data_file.seek(start)
        return read_ndarray(self.data_file)

    def _write_context_vector(self, array: np.ndarray):
        write_ndarray(self.data_file, array)
        self.data_file.flush()

    # Metadata

    def _read_metadata(self) -> CVMetadata:
        start = read_int64(self.metadata_file)
        end = read_int64(self.metadata_file)
        article_title = read_str(self.metadata_file)
        section_name = read_str(self.metadata_file)
        return CVMetadata(
            start,
            end,
            article_title,
            section_name,
        )

    def _write_metadata(self, metadata: CVMetadata):
        write_int64(self.metadata_file, metadata.start)
        write_int64(self.metadata_file, metadata.end)
        write_str(self.metadata_file, metadata.article_title)
        write_str(self.metadata_file, metadata.section_name)
        self.metadata_file.flush()

    # Metadata cache

    def _build_metadata_cache(self):
        self.metadata_cache.clear()
        self.metadata_file.seek(0)
        while True:
            try:
                metadata = self._read_metadata()
                self.metadata_cache.add(metadata)
            except EOFError:
                break

    def _cache_metadata(self, metadata: CVMetadata):
        self.metadata_cache.add(metadata)


# Serialization helpers


class EOFError(Exception):
    pass


def read_int64(file: BinaryIO) -> int:
    b = file.read(8)
    if len(b) < 8:
        raise EOFError()
    return struct.unpack("Q", b)[0]


def write_int64(file: BinaryIO, x: int):
    file.write(struct.pack("Q", x))


def read_int64s(file: BinaryIO) -> list[int]:
    b = read_bytes(file)
    return list(struct.unpack("Q" * (len(b) // 8), b))


def write_int64s(file: BinaryIO, x: list[int]):
    write_bytes(file, struct.pack("Q" * len(x), *x))


def read_str(file: BinaryIO) -> str:
    b = read_bytes(file)
    return b.decode()


def write_str(file: BinaryIO, s: str):
    b = s.encode()
    write_bytes(file, b)


def read_ndarray(file: BinaryIO) -> np.ndarray:
    dtype = read_str(file)
    shape = read_int64s(file)
    array_bytes = read_bytes(file)
    return np.frombuffer(array_bytes, dtype=dtype).reshape(shape)


def write_ndarray(file: BinaryIO, array: np.ndarray):
    write_str(file, str(array.dtype))
    write_int64s(file, array.shape)
    write_bytes(file, array.tobytes())


def read_bytes(file: BinaryIO) -> bytes:
    len_bytes = read_int64(file)
    return file.read(len_bytes)


def write_bytes(file: BinaryIO, b: bytes):
    file.write(struct.pack("Q", len(b)))
    file.write(b)
