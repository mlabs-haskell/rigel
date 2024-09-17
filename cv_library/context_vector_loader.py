import json
from pathlib import Path
import pickle
import random
import re
from typing import Iterator, Literal

import torch

class ContextVectorDataLoader():
    def __init__(
        self,
        batch_size: int,
        tfidf_file: str,
        split: Literal['train', 'val', 'test'],
        cv_dir_path: str,
        random_seed: int = 0
    ):
        # Get TFIDFs and document ids
        with open(tfidf_file, 'r') as file:
            tfidf_buffer = json.load(file)
            self.tfidfs = {}
            document_ids = []
            for document_id, tfidf in tfidf_buffer.items():
                self.tfidfs[document_id] = torch.tensor(tfidf)
                document_ids.append(document_id)

        # Determine batch selection function based on split type
        match split:
            case 'train':
                selection_function = lambda i: 0 <= i % 5 and i % 5 <= 2
            case 'val':
                selection_function = lambda i: i % 5 == 3
            case 'test':
                selection_function = lambda i: i % 5 == 4
            case _:
                raise ValueError(f"Unknown split type {split}")

        # Batch document ids
        random.seed(random_seed)
        random.shuffle(document_ids)
        batches = [
            document_ids[i : i + batch_size]
            for batch_number, i in enumerate(range(0, len(document_ids), batch_size))
            if selection_function(batch_number)
        ]

        self.batches = batches
        self.cv_dir_path = Path(cv_dir_path)

    def __len__(self):
        return len(self.batches)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.batches:
            X, y = self.load_batch(batch)
            yield X, y

    def load_batch(self, batch: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        Xs = []
        document_ids = []
        regex = re.compile(r"[^0-9a-zA-Z]")
        for document_id in batch:
            # Get file name and section heading
            article_title, section_header = document_id.split("\\", 1)
            filepath = self.cv_dir_path / (regex.sub("_", article_title) + ".pkl")

            # Find the document
            context_vector = None
            if filepath.is_file():
                with open(filepath, 'rb') as file:
                    doc = pickle.load(file)
                doc = doc["section_cv_map"]

                if section_header in doc:
                    context_vector = doc[section_header]

            if context_vector is None:
                print(f"Warning: {document_id} not found")
                continue
            if context_vector.shape[-2] != 256:
                continue
            Xs.append(context_vector)

            # Record the document id
            document_ids.append(document_id)
        X = torch.stack(Xs).to(torch.float32)

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

        return X, y
