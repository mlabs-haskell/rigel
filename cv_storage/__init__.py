from typing import IO, Iterable, Iterator
import numpy as np
import struct

import tqdm

class ContextVectorDB:
    """Class for saving and loading context vectors
    """
    def __init__(self, main_file: IO, metadata_file: IO):
        self.main_file = main_file
        self.metadata_file = metadata_file

    def append_context_vector(
        self,
        article_title: str,
        section_name: str,
        array: np.ndarray
    ) -> tuple[int, int]:
        """Adds a context vector to the database. Returns the start and end
        positions in the binary file
        """
        assert array.dtype == "float16", array.dtype

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
        self.metadata_file.write(struct.pack("Q", len(article_title)))
        self.metadata_file.write(article_title.encode())
        self.metadata_file.write(struct.pack("Q", len(section_name)))
        self.metadata_file.write(section_name.encode())
        return start, end

    def read_metadatum(self) -> tuple[str, str, int, int] | None:
        """Read the next item of metadata from the metadata file
        """
        start = self.metadata_file.read(8)
        if not start:
            return None
        start = struct.unpack("Q", start)[0]

        end = self.metadata_file.read(8)
        end = struct.unpack("Q", end)[0]

        article_title_length = self.metadata_file.read(8)
        article_title_length = struct.unpack("Q", article_title_length)[0]

        article_title = self.metadata_file.read(article_title_length)
        article_title = article_title.decode()

        section_name_length = self.metadata_file.read(8)
        section_name_length = struct.unpack("Q", section_name_length)[0]

        section_name = self.metadata_file.read(section_name_length)
        section_name = section_name.decode()

        return article_title, section_name, start, end

    def read_metadata(self, seek: int = 0) -> Iterator[tuple[str, str, int, int]]:
        """Iterate over the whole metadata file
        """
        if seek is not None:
            self.metadata_file.seek(seek)

        while True:
            if (metadatum := self.read_metadatum()) is not None:
                yield metadatum
            else:
                break

    def read_context_vectors(self) -> Iterator[tuple[str, str, np.ndarray]]:
        """Create an iterator that returns the next context vector. Each element
        in the iterator is a tuple of (article title, section name, context vector)
        """
        for article_title, section_name, start, _ in self.read_metadata():
            array = self.read_context_vector(start)
            yield article_title, section_name, array

    def read_context_vector(self, start: int) -> np.ndarray:
        """Get a context vector from the data file, starting from the given position
        """
        self.main_file.seek(start)
        shape_len = self.main_file.read(8)
        shape_len = struct.unpack("Q", shape_len)[0]

        shape = self.main_file.read(8 * shape_len)
        shape = struct.unpack("Q" * shape_len, shape)

        array = self.main_file.read(int(np.prod(shape) * 2))
        array = np.frombuffer(array, dtype="float16").reshape(shape)

        return array

class IndexedContextVectorDB:
    def __init__(self, main_file: str, metadata_file: str, progress: bool = True):
        """Open the metadata file, read it into memory and return an IndexedContextVectorDB."""
        with open(metadata_file, 'ab+') as metadata_f:
            with open(main_file, 'ab+') as main_f:
                db = ContextVectorDB(main_f, metadata_f)
                metadata_iter = db.read_metadata()
                if progress:
                    metadata_iter = tqdm.tqdm(metadata_iter)
            index_map = self.build_index(metadata_iter)

        self.index_map = index_map
        self.db = db

    @classmethod
    def build_index(
        cls,
        metadata: Iterator[tuple[str, str, int, int]]
    ) -> dict[str, dict[str, int]]:
        index_map = {}
        for article_title, section_name, start, _ in metadata:
            index_map.setdefault(article_title, {})
            index_map[article_title][section_name] = start
        return index_map

    def has_article(self, article_title: str) -> bool:
        """Helper function to determine if an article has been processed"""
        return article_title in self.index_map

    def get(self, article_title: str, section_name: str) -> np.ndarray | None:
        """Get the context vector for a given section in an article
        """
        if article_title not in self.index_map:
            return None
        if section_name not in self.index_map[article_title]:
            return None

        start = self.index_map[article_title][section_name]
        return self.db.read_context_vector(start)

    def insert(self, article_title: str, section_name: str, value: np.ndarray):
        start, _ = self.db.append_context_vector(article_title, section_name, value)
        self.index_map[article_title][section_name] = start

