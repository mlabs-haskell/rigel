import json
import random
import threading
from typing import Iterator

from bson.objectid import ObjectId
import torch

from database_utils import (
    CONNECTION_STRING, DATABASE_NAME, connect_to_database
)

class ContextVectorDataLoader():
    def __init__(
        self,
        batch_size: int,
        tfidf_file: str,
        connection_string: str = CONNECTION_STRING,
        database_name: str = DATABASE_NAME,
        random_seed: int | None = None,
        device: str | None = None
    ):
        # Get database connection
        database = connect_to_database(connection_string, database_name)

        # Get TFIDFs and document ids
        with open(tfidf_file, 'r') as file:
            tfidf_buffer = json.load(file)
            self.tfidfs = {}
            document_ids = []
            for collection_name, document_dict in tfidf_buffer.items():
                for document_id, tfidf in document_dict.items():
                    self.tfidfs[document_id] = torch.tensor(tfidf)
                    document_ids.append((collection_name, document_id))

        # Batch document ids
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(document_ids)
        batches = [
            document_ids[i : i + batch_size]
            for i in range(0, len(document_ids), batch_size)
        ]

        self.database = database
        self.batches = batches
        self.device = device

    def __len__(self):
        return len(self.batches)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.batches:
            X, y = self.load_batch(batch)
            yield X, y

    def load_batch(self, batch: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        Xs = []
        document_ids = []
        for collection_name, document_id in batch:
            # Find the document
            collection = self.database[collection_name]
            doc = collection.find_one({"_id": ObjectId(document_id)})
            if doc is None:
                continue

            # Get its context vector
            context_vector = doc["context_vector"]
            context_vector = torch.tensor(context_vector)
            if context_vector.shape[0] != 128 or context_vector.shape[1] != 4096:
                print(f"ALERT: {collection_name}, {document_id}")
                print(f"\tShape: {context_vector.shape}")
                print("\tDeleting...")
                collection.delete_one({'_id': ObjectId(document_id)})
                continue
            Xs.append(context_vector)

            # Record the document id
            document_ids.append(document_id)
        X = torch.stack(Xs)

        # Create y matrix containing similarity score between all data points
        cos_sim = torch.nn.CosineSimilarity(dim=0)
        y = torch.ones(len(batch), len(batch))
        for i in range(len(document_ids)):
            # Get ith document
            document_id_i = str(document_ids[i])
            tfidf_i = self.tfidfs[document_id_i]

            # Iterate through all future documents and calculate similarity score
            for j in range(i + 1, len(document_ids)):
                document_id_j = str(document_ids[j])
                tfidf_j = self.tfidfs[document_id_j]
                score = cos_sim(tfidf_i, tfidf_j)
                y[i, j] = y[j, i] = score

        # Make sure tensors are on right device
        if self.device is not None:
            X = X.to(self.device)
            y = y.to(self.device)

        return X, y