from pathlib import Path
import numpy as np

from indexed_binary_db import FileSpan, IndexedBinaryDB
from .models import CV, CVMetadata


class CVMetadataCache:
    def __init__(self):
        self.cache: dict[str, dict[str, FileSpan]] = {}

    def clear(self):
        self.cache.clear()

    def add(self, metadata: CVMetadata, file_span: FileSpan):
        title = metadata.article_title
        section = metadata.section_name
        if title not in self.cache:
            self.cache[title] = {}
        self.cache[title][section] = file_span

    def get(self, article_title: str, section_name: str) -> FileSpan | None:
        if article_title not in self.cache:
            return None
        return self.cache[article_title].get(section_name)

    def has_article(self, article_title: str) -> bool:
        return article_title in self.cache

    def get_article_titles(self) -> list[str]:
        return list(self.cache.keys())

    def get_section_names(self, article_title: str) -> list[str]:
        if article_title not in self.cache:
            return []
        return list(self.cache[article_title].keys())


class ContextVectorDB:
    """Class for saving and loading context vectors"""

    def __init__(self, folder: Path):
        if folder.exists():
            if not folder.is_dir():
                raise ValueError("Given path is not a folder: " + str(folder))
        else:
            folder.mkdir()
        index_file_path = folder / "index.cvdb"
        data_file_path = folder / "data.cvdb"

        self._db = IndexedBinaryDB(index_file_path, data_file_path, CVMetadata, CV)

        self.metadata_cache = CVMetadataCache()
        self._build_metadata_cache()

    # Public interface

    def get(self, article_title: str, section_name: str) -> np.ndarray | None:
        metadata = self.metadata_cache.get(article_title, section_name)
        if metadata is None:
            return None
        return self._read_context_vector(metadata.start)

    def has_article(self, article_title: str) -> bool:
        return self.metadata_cache.has_article(article_title)

    def has_section(self, article_title: str, section_name: str) -> bool:
        metadata = self.metadata_cache.get(article_title, section_name)
        return metadata is not None

    def get_article_titles(self) -> list[str]:
        return self.metadata_cache.get_article_titles()

    def get_section_names(self, article_title: str) -> list[str]:
        return self.metadata_cache.get_section_names(article_title)

    def insert(self, article_title: str, section_name: str, cv: np.ndarray):
        metadata = CVMetadata(article_title, section_name)
        file_span = self._db.write(metadata, CV(cv))
        self._cache_metadata(metadata, file_span)

    # Context vectors

    def _read_context_vector(self, start: int):
        cv: CV = self._db.read(start)
        return cv.cv

    # Metadata cache

    def _build_metadata_cache(self):
        self.metadata_cache.clear()
        index_entries = self._db.read_all_index_entries()
        for metadata, file_span in index_entries:
            self._cache_metadata(metadata, file_span)

    def _cache_metadata(self, metadata: CVMetadata, file_span: FileSpan):
        self.metadata_cache.add(metadata, file_span)
