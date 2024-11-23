import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, NamedTuple
import heapq

import torch

from indexed_binary_db import FileSpan, IndexedBinaryDB
from indexed_binary_db.reader import BinaryReader
from indexed_binary_db.writer import BinaryWriter

SimilarityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class DBConfig:
    level_sizes: list[int]


@dataclass
class Level:
    db: IndexedBinaryDB
    vec_size: int
    index: list[FileSpan]


class CVMetadata(NamedTuple):
    article_title: str
    section_name: str

    def write(self, writer: BinaryWriter):
        writer.write_str(self.article_title)
        writer.write_str(self.section_name)

    @classmethod
    def read(cls, reader: BinaryReader):
        article_title = reader.read_str()
        section_name = reader.read_str()
        return CVMetadata(article_title=article_title, section_name=section_name)


class CV(NamedTuple):
    cv: torch.Tensor

    def write(self, writer: BinaryWriter):
        writer.write_tensor(self.cv)

    @classmethod
    def read(cls, reader: BinaryReader):
        return CV(reader.read_tensor())


class SearchResult(NamedTuple):
    idx: int
    score: float
    cv: torch.Tensor


def parse_db_index(index: list[tuple[Any, FileSpan]]) -> list[FileSpan]:
    return [file_span for _, file_span in index]


class ContextVectorHierDB:
    def __init__(self, folder: Path, config: DBConfig, similarity_fn: SimilarityFn):
        self.similarity_fn = similarity_fn
        self.config = config
        self.folder = folder

        self.config_file = folder / "config.json"
        check_config_file(self.config_file, config)

        self._metadata_db = IndexedBinaryDB(
            index_path=folder / "metadata_index.db",
            data_path=folder / "metadata.db",
            metadata_cls=None,
            obj_cls=CVMetadata,
        )

        self._metadata: list[CVMetadata] = self._metadata_db.read_all()

        self.levels: list[Level] = []
        for level_size in config.level_sizes:
            index_path = folder / f"level_{level_size}_index.db"
            data_path = folder / f"level_{level_size}.db"
            db = IndexedBinaryDB(
                index_path=index_path,
                data_path=data_path,
                metadata_cls=None,
                obj_cls=CV,
            )
            index = parse_db_index(db.read_all_index_entries())

            self.levels.append(
                Level(
                    db=db,
                    index=index,
                    vec_size=level_size,
                )
            )

    # Public API

    def insert(
        self,
        metadata: CVMetadata,
        vecs: list[torch.Tensor],
    ):
        """
        vecs:
            list of ndarray of shape (seq_len, level.vec_size)
            seq_len can be anything.
            len(vecs) must be len(self.levels)
        """
        self._check_vec_sizes(vecs)

        self._metadata_db.write(None, metadata)
        self._metadata.append(metadata)
        for vec, level in zip(vecs, self.levels):
            level.index.append(level.db.write(None, CV(vec)))

    def _search_level(
        self,
        level_idx: int,
        query: torch.Tensor,
        previous_results: list[SearchResult] | None,
        narrow_factor: int,
    ) -> list[SearchResult]:
        if previous_results is None:
            haystack = list(enumerate(self._read_level(0)))
        else:
            haystack = [
                (v.idx, self._read_level_vec(level_idx, v.idx))
                for v in previous_results
            ]
        result_size = len(haystack) // narrow_factor
        assert result_size > 0
        return get_top_k_similar(
            query,
            haystack,
            self.similarity_fn,
            result_size,
        )

    def search(
        self,
        query: list[torch.Tensor],
        narrow_factor: int,
        max_level: int | None = None,
    ) -> list[SearchResult]:
        """
        query:
            List of context vectors corresponding to each level.
            Must have shape: (seq_len, level.vec_size). seq_len can be anything.
        narrow_factor:
            At each level, take N/narrow_factor many context vectors to the next level.
            N is the size of the result set at that level.
            N = size of the entire DB at level 0.
        max_level:
            If not None, stop descending to levels after this level.
            For example, if max_level = 0, only search the first level.
        """
        self._check_vec_sizes(query)

        if max_level is None:
            max_level = len(self.levels) - 1

        results = None
        for level_idx, q in zip(range(max_level + 1), query):
            results = self._search_level(level_idx, q, results, narrow_factor)

        assert results is not None
        return results

    def get_metadata(self, idx: int) -> CVMetadata:
        return self._metadata[idx]

    # Internals

    def _read_level(self, level_idx: int) -> list[torch.Tensor]:
        level = self.levels[level_idx]
        return [cv.cv for cv in level.db.read_all()]

    def _read_level_vec(self, level_idx: int, idx: int) -> torch.Tensor:
        """Read the vector at index idx from the level"""
        level = self.levels[level_idx]
        index = level.index[idx]
        cv: CV = level.db.read(index.start)
        return cv.cv

    def _check_vec_sizes(self, vecs: list[torch.Tensor]):
        """Ensure the shape of a vector at each level is [seq_len, level.vec_size]"""
        assert len(vecs) == len(self.levels), f"{len(vecs)} != {len(self.levels)}"
        for vec, level in zip(vecs, self.levels):
            assert vec.shape[-1] == level.vec_size


def check_config_file(path: Path, config: DBConfig):
    if not path.exists():
        path.write_text(json.dumps(config.__dict__))
    else:
        with open(path, "r") as file:
            file_config = json.load(file)
        expected_config = dataclasses.asdict(config)
        if file_config != expected_config:
            raise ValueError(
                f"Config file {path} does not match expected config: {config} != {file_config}"
            )


def get_top_k_similar(
    query: torch.Tensor,
    haystack: list[tuple[int, torch.Tensor]],  # list of (idx, cv)
    similarity_fn: SimilarityFn,
    k: int,
) -> list[SearchResult]:
    # Heap of (score, idx)
    heap: list[tuple[float, SearchResult]] = []

    vecs_by_len: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for idx, v in haystack:
        seq_len = v.shape[0]
        if seq_len not in vecs_by_len:
            vecs_by_len[seq_len] = []
        vecs_by_len[seq_len].append((idx, v))

    for seq_len, vecs_of_len in vecs_by_len.items():
        batch = torch.stack([v for _, v in vecs_of_len])
        scores = similarity_fn(query, batch)
        for (idx, cv), score in zip(vecs_of_len, scores):
            score_float = float(score)
            heapq.heappush(heap, (-score_float, SearchResult(idx, score_float, cv)))

    result: list[SearchResult] = []
    for _ in range(k):
        _, res = heapq.heappop(heap)
        result.append(res)

    return result
