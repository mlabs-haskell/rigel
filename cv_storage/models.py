import torch

from typing import NamedTuple

from .indexed_binary_db.reader import BinaryReader
from .indexed_binary_db.writer import BinaryWriter


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
