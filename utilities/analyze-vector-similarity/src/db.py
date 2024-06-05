import os
from typing import TypedDict
from dumps import Doc
import json
import numpy as np


class DBDoc(TypedDict):
    id: str
    title: str
    header_name: str
    context_vectors_offset: int
    context_vectors_len: int


class DBWriter:
    def __init__(self, dirname: str):
        os.makedirs(dirname, exist_ok=False)
        self.context_vectors = open(dirname + "/context_vectors.db", "wb")
        self.documents = open(dirname + "/documents.db", "w")

    def write_doc(self, doc: Doc):
        buffer = np.array(doc["context_vector"], dtype=np.float64).flatten().tobytes()
        db_doc: DBDoc = {
            "id": str(doc["_id"]),
            "title": doc["title"],
            "header_name": doc["header_name"],
            "context_vectors_offset": self.context_vectors.tell(),
            "context_vectors_len": len(buffer),
        }
        self.context_vectors.write(buffer)
        self.documents.write(json.dumps(db_doc))
        self.documents.write("\n")


class DBReader:
    def __init__(self, dirname: str):
        self.context_vectors = open(dirname + "/context_vectors.db", "rb")
        self.documents = open(dirname + "/documents.db", "r")

    def read_docs(self) -> list[DBDoc]:
        return [json.loads(line) for line in self.documents.readlines()]

    def get_context_vectors(self, doc: DBDoc) -> np.ndarray:
        self.context_vectors.seek(doc["context_vectors_offset"])
        length = doc["context_vectors_len"]
        buffer = self.context_vectors.read(length)
        return np.frombuffer(buffer, np.float64)
