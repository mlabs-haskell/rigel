import json
import random
from typing import Iterator

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
        random_seed: int | None = None
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
        document_ids = random.shuffle(document_ids)
        batches = [
            document_ids[i : i + batch_size]
            for i in range(0, len(document_ids), batch_size)
        ]

        self.database = database
        self.batches = batches

    def __next__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.batches:
            Xs = []
            document_ids = []
            for collection_name, document_id in batch:
                # Get the context vector
                doc = self.database[collection_name].find_one({"_id": document_id})
                context_vector = doc["context_vector"]
                context_vector = torch.tensor(context_vector)
                context_vector = context_vector.flatten()
                Xs.append(context_vector)

                # Record the document id
                document_ids.append(document_id)
            X = torch.stack(Xs)

            # Create y matrix containing similarity score between all data points
            cos_sim = torch.nn.CosineSimilarity(dim=0)
            y = torch.ones(len(batch), len(batch))
            for i in range(len(document_ids)):
                document_id_i = str(document_ids[i])
                tfidf_i = self.tfidfs[document_id_i]
                for j in range(i + 1, len(document_ids)):
                    document_id_j = str(document_ids[j])
                    tfidf_j = self.tfidfs[document_id_j]
                    score = cos_sim(tfidf_i, tfidf_j)
                    y[i, j] = y[j, i] = score

            yield X, y