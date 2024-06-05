import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import scipy.stats
import json

import db
import config


def main():
    dbr = db.DBReader(config.DB_DIR)
    groups_dir = Path(config.GROUPS_DIR)
    group_files = sorted((x for x in groups_dir.iterdir()), key=lambda x: x.name)

    groups_map = {}
    for group in group_files:
        file = open(group)
        groups_map[group.name] = [x.strip() for x in file.readlines()]

    docs = dbr.read_docs()
    docs = [d for d in docs if d["id"] not in config.EXCLUDE_IDS]
    print(len(docs))

    output = {}

    for group, titles in groups_map.items():
        print(group)

        group_output = {}
        output[group] = group_output

        group_docs = [doc for doc in docs if doc["title"] in titles]
        print(" ", len(group_docs), "docs")

        sample = random.sample(group_docs, config.SAMPLE_SIZE)

        si = similarity_index_symmetric(dbr, sample)
        print(" ", "SI:", si)
        group_output["self_similarity_index"] = si.asdict()

        group_output["cross_similarity_index"] = {}
        for group2, titles2 in groups_map.items():
            if group2 == group:
                continue
            print("   ", group2)
            group2_docs = [doc for doc in docs if doc["title"] in titles2]
            sample2 = random.sample(group2_docs, config.SAMPLE_SIZE)
            si = similarity_index(dbr, sample, sample2)
            group_output["cross_similarity_index"][group2] = si.asdict()
            print("     ", "SI:", si)

    os.makedirs(config.OUT_DIR, exist_ok=True)
    json.dump(output, open(config.OUT_DIR + "/similarity-index.json", "w"))


class Metrics:
    def __init__(self, arr: np.ndarray):
        arr = arr.flatten()
        self.mean = arr.mean()
        self.gmean = scipy.stats.gmean(arr)
        self.min = arr.min()
        self.max = arr.max()
        self.std = arr.std()

    def __str__(self):
        return "AM {:.4f}; GM {:.4f}; Min {:.4f}; Max {:.4f}; Std {:.4f}".format(
            self.mean, self.gmean, self.min, self.max, self.std
        )

    def asdict(self) -> dict:
        return {
            "mean": self.mean,
            "gmean": self.gmean,
            "min": self.min,
            "max": self.max,
            "std": self.std,
        }


def similarity_index_symmetric(dbr: db.DBReader, sample: list[db.DBDoc]):
    vectors = [dbr.get_context_vectors(doc) for doc in sample]
    similarity_matrix = cosine_similarity(vectors, vectors)
    # because X = Y, remove diagonal and symmetric elements
    # keep only one of M(i,j), M(j,i), remove M(i,i)
    similarity_matrix = np.triu(similarity_matrix, k=1)
    elements = similarity_matrix[np.nonzero(similarity_matrix)]
    return Metrics(elements)


def similarity_index(
    dbr: db.DBReader, sample1: list[db.DBDoc], sample2: list[db.DBDoc]
):
    vectors1 = [dbr.get_context_vectors(doc) for doc in sample1]
    vectors2 = [dbr.get_context_vectors(doc) for doc in sample2]
    similarity_matrix = cosine_similarity(vectors1, vectors2)
    return Metrics(similarity_matrix)


if __name__ == "__main__":
    main()
