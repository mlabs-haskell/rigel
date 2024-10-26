import struct
from typing import BinaryIO

import numpy as np


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
