import json
import random
from typing import Iterator, Literal

import torch

from cv_storage import ContextVectorDB


class ContextVectorDataLoader:
    def __init__(
        self,
        batch_size: int,
        tfidf_file: str,
        split: Literal["train", "val", "test"],
        cvdb_folder: str,
        random_seed: int = 0,
    ):
        # Get TFIDFs and article titles
        # Schema:
        # {
        #   "article_title": {
        #     "section_name": [tf-idf values],
        #   },
        # }
        with open(tfidf_file, "r") as file:
            self.tfidfs: dict[str, dict[str, list[int]]] = json.load(file)

        # Determine batch selection function based on split type
        match split:
            case "train":
                selection_function = lambda i: 0 <= i % 5 and i % 5 <= 2
            case "val":
                selection_function = lambda i: i % 5 == 3
            case "test":
                selection_function = lambda i: i % 5 == 4
            case _:
                raise ValueError(f"Unknown split type {split}")

        article_keys = []
        for article_title, sections in self.tfidfs.items():
            for section_name in sections:
                article_keys.append((article_title, section_name))

        # Batch article titles
        random.seed(random_seed)
        random.shuffle(article_keys)
        batches = [
            article_keys[i : i + batch_size]
            for batch_number, i in enumerate(range(0, len(article_keys), batch_size))
            if selection_function(batch_number)
        ]

        self.cv_db = ContextVectorDB(cvdb_folder)
        # List of batches, where each batch is a list of (article_title, section_name)
        self.batches: list[list[tuple[str, str]]] = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.batches:
            X, y = self.load_batch(batch)
            yield X, y

    def load_batch(
        self, batch: list[tuple[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Xs = []
        article_keys = []
        for article_title, section_name in batch:
            # Get the context vector
            context_vector = self.cv_db.get(article_title, section_name)
            context_vector = torch.tensor(context_vector)
            # If not the right size, skip
            # We can't remove it because the ContextVectorDB doesn't support deletion yet
            if context_vector.shape[0] != 128 or context_vector.shape[1] != 4096:
                print(f"ALERT: {article_title}, {section_name}")
                print(f"\tShape: {context_vector.shape}")
                print("\tSkipping...")
                continue
            Xs.append(context_vector)

            # Record the document id
            article_keys.append((article_title, section_name))
        X = torch.stack(Xs)

        # Create y matrix containing similarity score between all data points
        cos_sim = torch.nn.CosineSimilarity(dim=0)
        y = torch.ones(len(batch), len(batch))
        for i in range(len(article_keys)):
            # Get ith document
            article_title_i, section_name_i = article_keys[i]
            tfidf_i = self.tfidfs[article_title_i][section_name_i]

            # Iterate through all future documents and calculate similarity score
            for j in range(i + 1, len(article_keys)):
                article_title_j, section_name_j = article_keys[j]
                tfidf_j = self.tfidfs[article_title_j][section_name_j]
                score = cos_sim(tfidf_i, tfidf_j)
                y[i, j] = y[j, i] = score

        return X, y
