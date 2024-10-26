from dataclasses import dataclass
import json
from os import write
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np

from bin_storage import read_int64, read_ndarray, write_int64, write_ndarray

SimilarityFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class DBConfig:
    vec_max_size: int
    vec_min_size: int
    compression_factor: int
    search_narrow_factor: int
    # compress only shape[copression_dimension] of the vector
    compression_dimension: int


@dataclass
class Level:
    idx: int
    data_file: BinaryIO
    index_file: BinaryIO
    vec_size: int


class Database:
    def __init__(self, folder: Path, config: DBConfig, similariy_fn: SimilarityFn):
        self.similarity_fn = similariy_fn
        self.config = config
        self.folder = folder

        self.config_file = folder / "config.json"
        check_config_file(self.config_file, config)

        self.levels: list[Level] = []
        for i, level_size in enumerate(calculate_level_sizes(config)):
            data_file_path = folder / f"level_{level_size}.db"
            data_file = data_file_path.open("a+b")
            index_file_path = folder / f"level_{level_size}_index.db"
            index_file = index_file_path.open("a+b")
            self.levels.append(
                Level(
                    idx=i,
                    data_file=data_file,
                    index_file=index_file,
                    vec_size=level_size,
                )
            )

    def __del__(self):
        for level in self.levels:
            level.index_file.close()
            level.data_file.close()

    # Public API

    def insert(self, vecs: list[np.ndarray]):
        """We expect the shape of a vector to be [seq_len, level.vec_size]"""
        self._check_vec_sizes(vecs)

        for vec, level in zip(vecs, self.levels):
            self._insert_at_level(level, vec)

    def search(self, query_vecs: list[np.ndarray]) -> list[np.ndarray]:
        """We expect the shape of a query to be [seq_len, level.vec_size]"""
        self._check_vec_sizes(query_vecs)

        level_iter = iter(self.levels)
        query_iter = iter(query_vecs)

        # Read the first level and query
        level = next(level_iter)
        query = next(query_iter)

        # Read all the vectors from the first level
        vecs = self._read_level(level)

        # Calculate the indices of the most similar vectors
        ranks = rank_by_similarity(
            query,
            vecs,
            self.similarity_fn,
        )
        next_search_size = len(vecs) // self.config.search_narrow_factor
        ranks = ranks[:next_search_size]

        while True:
            try:
                # Get the next level and query
                level = next(level_iter)
                query = next(query_iter)
            except StopIteration:
                return [vecs[idx] for idx in ranks]

            # Use the most similar indices from the previous level to narrow down the search
            vecs = [self._read_level_vec(level, idx) for idx in ranks]
            ranks = rank_by_similarity(
                query,
                vecs,
                self.similarity_fn,
            )
            next_search_size = len(vecs) // self.config.search_narrow_factor
            ranks = ranks[:next_search_size]

    # Internals

    def _insert_at_level(self, level: Level, vec: np.ndarray):
        # Write data
        start = level.data_file.tell()
        write_ndarray(level.data_file, vec)
        end = level.data_file.tell()
        # Write index
        write_int64(level.index_file, start)
        write_int64(level.index_file, end)

    def _read_level(self, level: Level) -> list[np.ndarray]:
        vecs = []
        level.data_file.seek(0)
        while True:
            try:
                vec = read_ndarray(level.data_file)
            except EOFError:
                break
            vecs.append(vec)
        return vecs

    def _read_level_index(self, level: Level) -> list[tuple[int, int]]:
        res = []
        level.index_file.seek(0)
        while True:
            try:
                start = read_int64(level.index_file)
                end = read_int64(level.index_file)
            except EOFError:
                break
            res.append((start, end))
        return res

    def _read_level_vec(self, level: Level, idx: int) -> np.ndarray:
        """Read the vector at index idx from the level"""
        index = self._read_level_index(level)[idx]
        level.data_file.seek(index[0])
        return read_ndarray(level.data_file)

    def _check_vec_sizes(self, vecs: list[np.ndarray]):
        """Ensure the shape of a vector at each level is [seq_len, level.vec_size]"""
        assert len(vecs) == len(self.levels)
        for vec, level in zip(vecs, self.levels):
            assert vec.shape[self.config.compression_dimension] == level.vec_size


def check_config_file(path: Path, config: DBConfig):
    if not path.exists():
        path.write_text(json.dumps(config.__dict__))
    else:
        with open(path, "r") as file:
            file_config = json.load(file)
        if file_config != config:
            raise ValueError(
                f"Config file {path} does not match expected config: {config} != {file_config}"
            )


def calculate_level_sizes(config: DBConfig) -> list[int]:
    # Start with min_size and multiply with compression factor until max_size is hit
    # If it's not exactly equal to max_size, throw invalid config error
    res = []
    size = config.vec_min_size
    while size <= config.vec_max_size:
        res.append(size)
        size *= config.compression_factor
    if res[-1] != config.vec_max_size:
        raise ValueError(
            f"Invalid configuration: {config.vec_max_size} is not reachable from {config.vec_min_size} "
            f"using compression factor {config.compression_factor}"
        )
    return res


def rank_by_similarity(
    query: np.ndarray,
    vecs: list[np.ndarray],
    similarity_fn: SimilarityFn,
) -> list[int]:
    vecs_similarity = []
    for i, vec in enumerate(vecs):
        sim = similarity_fn(query, vec)
        vecs_similarity.append((i, sim))
    vecs_similarity.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in vecs_similarity]


class DBCorruptError(Exception):
    pass
