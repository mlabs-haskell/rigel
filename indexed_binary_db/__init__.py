from pathlib import Path
from typing import Any, NamedTuple

from .reader import BinaryReader
from .writer import BinaryWriter


class FileSpan(NamedTuple):
    start: int
    len: int


class IndexedBinaryDB:
    def __init__(
        self,
        index_path: Path,
        data_path: Path,
        metadata_cls,
        obj_cls,
    ):
        """
        metadata_cls:
            Must support cls.read(reader: BinaryReader) and self.write(writer: BinaryWriter)
        obj_cls:
            Must support cls.read(reader: BinaryReader) and self.write(writer: BinaryWriter)
        """
        self._index_file = index_path.open("a+b")
        self._index_reader = BinaryReader(self._index_file)
        self._index_writer = BinaryWriter(self._index_file)

        self._data_file = data_path.open("a+b")
        self._data_reader = BinaryReader(self._data_file)
        self._data_writer = BinaryWriter(self._data_file)

        self._metadata_cls = metadata_cls
        self._obj_cls = obj_cls

    def __del__(self):
        if hasattr(self, "_index_file"):
            self._index_file.close()
        if hasattr(self, "_data_file"):
            self._data_file.close()

    def read_all(self) -> list[Any]:
        self._data_file.seek(0)
        res = []
        while True:
            try:
                obj = self._obj_cls.read(self._data_reader)
            except EOFError:
                break
            res.append(obj)
        return res

    def read(self, start: int):
        self._data_file.seek(start)
        return self._obj_cls.read(self._data_reader)

    def write(self, metadata, obj) -> FileSpan:
        assert (
            # metadata and metadata_cls are both present
            (metadata is None and self._metadata_cls is None)
            or
            # metadata and metadata_cls are both absent
            (metadata is not None and self._metadata_cls is not None)
        )
        start = self._data_file.tell()
        obj.write(self._data_writer)
        end = self._data_file.tell()
        len_ = end - start
        file_span = FileSpan(start, len_)
        self.write_index_entry(metadata, file_span)
        return file_span

    def flush(self):
        self._data_file.flush()
        self._index_file.flush()

    def read_all_index_entries(self) -> list[tuple[Any, FileSpan]]:
        """
        Returns list[tuple[Metadata, FileSpan]].
        """
        self._index_file.seek(0)
        res: list[tuple[Any, FileSpan]] = []
        while True:
            try:
                metadata, file_span = self.read_index_entry()
            except EOFError:
                break
            res.append((metadata, file_span))
        return res

    def write_index_entry(self, metadata, file_span: FileSpan) -> None:
        self._index_writer.write_int64(file_span.start)
        self._index_writer.write_int64(file_span.len)
        if metadata is not None:
            metadata.write(self._index_writer)

    def read_index_entry(self) -> tuple[Any, FileSpan]:
        start = self._index_reader.read_int64()
        len_ = self._index_reader.read_int64()
        file_span = FileSpan(start, len_)

        metadata = None
        if self._metadata_cls is not None:
            metadata = self._metadata_cls.read(self._index_reader)

        return (metadata, file_span)
