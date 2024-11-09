class Buffer:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, nbytes: int):
        res = self._data[self._pos : self._pos + nbytes]
        self._pos += len(res)
        return res
