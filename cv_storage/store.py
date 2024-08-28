import numpy as np
import io
import struct


class ContextVectorDB:

    def __init__(self, main_file, metadata_file):
        self.main_file = main_file
        self.metadata_file = metadata_file

    def append_context_vector(self, title: str, array: np.ndarray):
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

    def read_metadata(self, seek=0):
        if seek is not None:
            self.metadata_file.seek(seek)

        while True:
            start = self.metadata_file.read(8)
            if not start:
                break
            start = struct.unpack("Q", start)[0]

            end = self.metadata_file.read(8)
            end = struct.unpack("Q", end)[0]

            title_length = self.metadata_file.read(8)
            title_length = struct.unpack("Q", title_length)[0]

            title = self.metadata_file.read(title_length)
            title = title.decode()

            yield title, (start, end)

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
