import json
from pathlib import Path
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
        skip_small_batches: bool = True
    ):
        # Get TFIDFs and article titles
        # Schema:
        # {
        #   "article_title": {
        #     "section_name": [tf-idf values],
        #   },
        # }
        with open(tfidf_file, "r") as file:
            tfidf_dict: dict[str, dict[str, dict]] = json.load(file)
            self.tfidfs: dict[int, dict[str, dict[str, list[float]]]] = {}
            for article_title, sections in tfidf_dict.items():
                for section_name, data in sections.items():
                    seq_len = int(data["seq_len"])
                    tfidf = data["tfidf"]

                    self.tfidfs.setdefault(seq_len, {})
                    self.tfidfs[seq_len].setdefault(article_title, {})
                    self.tfidfs[seq_len][article_title][section_name] = torch.tensor(tfidf)

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

        # Get the list of keys for each batch
        article_keys = {}
        for seq_len, articles in self.tfidfs.items():
            article_keys[seq_len] = [
                (seq_len, article_title, section_name)
                for article_title, sections in articles.items()
                for section_name in sections
            ]

        # Batch article titles
        random.seed(random_seed)
        batches = []
        counter = 0
        for keys in article_keys.values():
            random.shuffle(keys)
            for i in range(0, len(keys), batch_size):
                if selection_function(counter):
                    batch = keys[i: i + batch_size]
                    if len(batch) == batch_size or not skip_small_batches:
                        batches.append(keys[i: i + batch_size])
                counter += 1

        cvdb_folder = Path(cvdb_folder)
        self.cv_db = ContextVectorDB(cvdb_folder)
        # List of batches, where each batch is a list of (article_title, section_name)
        self.batches: list[list[tuple[int, str, str]]] = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.batches:
            X, y = self.load_batch(batch)
            yield X, y

    def load_batch(
        self, batch: list[tuple[int, str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Xs = []
        for _, article_title, section_name in batch:
            # Get the context vector
            context_vector = self.cv_db.get(article_title, section_name)
            context_vector = torch.tensor(context_vector)
            Xs.append(context_vector)
        X = torch.stack(Xs).to(torch.float32)

        # Create y matrix containing similarity score between all data points
        cos_sim = torch.nn.CosineSimilarity(dim=0)
        y = torch.ones(len(batch), len(batch))
        for i in range(len(batch)):
            # Get ith document
            seq_len_i, article_title_i, section_name_i = batch[i]
            tfidf_i = self.tfidfs[seq_len_i][article_title_i][section_name_i]

            # Iterate through all future documents and calculate similarity score
            for j in range(i + 1, len(batch)):
                seq_len_j, article_title_j, section_name_j = batch[j]
                tfidf_j = self.tfidfs[seq_len_j][article_title_j][section_name_j]
                score = cos_sim(tfidf_i, tfidf_j)
                y[i, j] = y[j, i] = score

        return X, y
