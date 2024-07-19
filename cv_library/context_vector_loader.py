from pymongo import MongoClient
import random
import torch
from typing import Iterator

CONNECTION_STRING = "mongodb://127.0.0.1"
DATABASE_NAME = "rigel"

class ContextVectorDataLoader():
    def __init__(
        self,
        batch_size: int,
        connection_string: str = CONNECTION_STRING,
        database_name: str = DATABASE_NAME,
        random_seed: int | None = None
    ):
        # Get all document ids
        client = MongoClient(connection_string)
        database = client[database_name]
        document_ids = []
        for collection_name in database.list_collection_names():
            collection = database[collection_name]
            for document in collection.find():
                document_ids.append((collection_name, document["_id"]))

        # Batch them
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
            article_list = []
            for collection_name, document_id in batch:
                doc = self.database[collection_name].find({"_id": document_id})
                context_vector = doc["context_vector"]
                context_vector = torch.tensor(context_vector)
                context_vector = context_vector.flatten()
                Xs.append(context_vector)

                title = doc["title"]
                header_name = doc["header_name"]
                article_list.append((title, header_name))

            X = torch.stack(Xs)