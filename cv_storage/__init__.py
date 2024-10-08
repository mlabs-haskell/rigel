from typing import Iterable
import numpy as np
import struct

import tqdm


class ContextVectorDB:

    def __init__(self, main_file, metadata_file):
        self.main_file = main_file
        self.metadata_file = metadata_file

    def append_context_vector(self, title: str, array: np.ndarray) -> tuple[int, int]:
        assert array.dtype == "float64", array.dtype

        start = self.main_file.tell()

        # write the length of the shape of the array (u64)
        # followed by the shape of the array (u64)
        # followed by the array itself (numpy.to_bytes(), f64)
        self.main_file.write(struct.pack("Q", len(array.shape)))
        self.main_file.write(struct.pack("Q" * len(array.shape), *array.shape))
        self.main_file.write(array.tobytes())

        end = self.main_file.tell()

        # write start: u64, end: u64, length of title: u64, title: str
        # Note: We don't _need_ to write `end` into the metadata file, because
        # `shape` determines the length of the context vector.
        # But it's still useful to have `end` in the metadata file so
        # that we can slice up the `main_file` without having to read the
        # `main_file` to read `shape`. This could help process the context
        # vectors in parallel.
        self.metadata_file.write(struct.pack("Q", start))
        self.metadata_file.write(struct.pack("Q", end))
        self.metadata_file.write(struct.pack("Q", len(title)))
        self.metadata_file.write(title.encode())
        return start, end

    def read_metadatum(self):
        start = self.metadata_file.read(8)
        if not start:
            return None
        start = struct.unpack("Q", start)[0]

        end = self.metadata_file.read(8)
        end = struct.unpack("Q", end)[0]

        title_length = self.metadata_file.read(8)
        title_length = struct.unpack("Q", title_length)[0]

        title = self.metadata_file.read(title_length)
        title = title.decode()
        return title, (start, end)

    def read_metadata(self, seek=0):
        if seek is not None:
            self.metadata_file.seek(seek)

        while True:
            if (metadatum := self.read_metadatum()) is not None:
                yield metadatum
            else:
                break

    def read_context_vectors(self):
        for title, (start, _end) in self.read_metadata():
            array = self.read_context_vector(start)
            yield title, array

    def read_context_vector(self, start):
        self.main_file.seek(start)
        shape_len = self.main_file.read(8)
        shape_len = struct.unpack("Q", shape_len)[0]

        shape = self.main_file.read(8 * shape_len)
        shape = struct.unpack("Q" * shape_len, shape)

        array = self.main_file.read(int(np.prod(shape) * 8))
        array = np.frombuffer(array, dtype="float64").reshape(shape)

        return array


Index = dict[str, int]


class IndexedContextVectorDB:
    def __init__(self, index: Index, db: ContextVectorDB):
        self.index = index
        self.db = db

    @classmethod
    def open(cls, main_file, metadata_file, progress=True):
        """Open the metadata file, read it into memory and return an IndexedContextVectorDB."""
        with open(metadata_file) as metadata_f:
            with open(main_file) as main_f:
                db = ContextVectorDB(main_f, metadata_f)
                metadata_iter = db.read_metadata()
                if progress:
                    metadata_iter = tqdm.tqdm(metadata_iter)
            index = cls.build_index(metadata_iter)

        return cls(index, ContextVectorDB(open(main_file), open(metadata_file)))

    @classmethod
    def build_index(cls, metadata: Iterable[tuple[str, tuple[int, int]]]):
        index = {}
        for key, (start, _end) in metadata:
            index[key] = start
        return index

    def get(self, key: str) -> np.ndarray | None:
        if key not in self.index:
            return None
        start = self.index[key]
        return self.db.read_context_vector(start)

    def insert(self, key: str, value: np.ndarray):
        start, _end = self.db.append_context_vector(key, value)
        self.index[key] = start

