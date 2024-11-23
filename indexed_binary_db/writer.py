from typing import BinaryIO
import struct
import torch


class BinaryWriter:
    def __init__(self, file: BinaryIO):
        self._file = file

    def read_int64(self) -> int:
        b = self._file.read(8)
        if len(b) < 8:
            raise EOFError()
        return struct.unpack("Q", b)[0]

    def write_int64(self, x: int):
        self._file.write(struct.pack("Q", x))

    def write_int64s(self, x: list[int]):
        self.write_bytes(struct.pack("Q" * len(x), *x))

    def write_str(self, s: str):
        b = s.encode()
        self.write_bytes(b)

    def write_tensor(self, tensor: torch.Tensor):
        array = tensor.numpy()
        self.write_str(str(array.dtype))
        self.write_int64s(array.shape)
        self.write_bytes(array.tobytes())

    def write_bytes(self, b: bytes):
        self._file.write(struct.pack("Q", len(b)))
        self._file.write(b)
