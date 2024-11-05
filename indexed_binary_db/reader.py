import struct
from typing import BinaryIO

import numpy as np

from .buffer import Buffer


class BinaryReader:
    def __init__(self, file: BinaryIO | Buffer):
        self._file = file

    def read_int64(self) -> int:
        b = self._file.read(8)
        if len(b) < 8:
            raise EOFError()
        return struct.unpack("Q", b)[0]

    def read_int64s(self) -> list[int]:
        b = self.read_bytes()
        return list(struct.unpack("Q" * (len(b) // 8), b))

    def read_str(self) -> str:
        b = self.read_bytes()
        return b.decode()

    def read_ndarray(self) -> np.ndarray:
        dtype = self.read_str()
        shape = self.read_int64s()
        array_bytes = self.read_bytes()
        return np.frombuffer(array_bytes, dtype=dtype).reshape(shape)

    def read_bytes(self) -> bytes:
        len_bytes = self.read_int64()
        return self._file.read(len_bytes)
