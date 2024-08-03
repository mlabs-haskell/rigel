import random
from typing import Iterator

import torch

from .database_utils import (
    CONNECTION_STRING, DATABASE_NAME,
    connect_to_database, get_contents_indices, get_documents_from_mongo, read_data
)

class ContextVectorDataLoader():
    def __init__(
        self,
        batch_size: int,
        contents_data_file: str,
        contents_index_file: str,
        connection_string: str = CONNECTION_STRING,
        database_name: str = DATABASE_NAME,
        random_seed: int | None = None,
        vocabulary: dict[str, int] = None
    ):
        # Get documents and database connection
        database = connect_to_database(connection_string, database_name)
        document_ids = list(get_documents_from_mongo(database))

        # Batch them
        if random_seed is not None:
            random.seed(random_seed)
        document_ids = random.shuffle(document_ids)
        batches = [
            document_ids[i : i + batch_size]
            for i in range(0, len(document_ids), batch_size)
        ]

        self.contents_indices = get_contents_indices(contents_index_file)
        self.contents_data_file = contents_data_file
        self.database = database
        self.batches = batches
        self.vocabulary = vocabulary

    def __next__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.batches:
            Xs = []
            article_list = []
            for collection_name, document_id in batch:
                # Get the context vector
                doc = self.database[collection_name].find_one({"_id": document_id})
                context_vector = doc["context_vector"]
                context_vector = torch.tensor(context_vector)
                context_vector = context_vector.flatten()
                Xs.append(context_vector)

                # Get the document info
                title = doc["title"]
                header_name = doc["header_name"]
                article_list.append((title, header_name))

            # Create y matrix containing similarity score between all data points
            y = torch.ones(len(batch), len(batch))
            for i in range(len(article_list)):
                title_i, header_name_i = article_list[i]
                index_i = self.contents_indices[title_i]
                article_i = read_data(index_i)
                text_i = article_i[header_name_i]

            X = torch.stack(Xs)