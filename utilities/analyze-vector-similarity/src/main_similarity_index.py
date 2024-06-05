import db
import config

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np


def main():
    dbr = db.DBReader("db")
    groups_dir = Path("groups")
    group_files = list(x for x in groups_dir.iterdir())

    groups_map = {}
    for group in group_files:
        file = open(group)
        groups_map[group.name] = [x.strip() for x in file.readlines()]

    docs = dbr.read_docs()
    docs = [d for d in docs if d["id"] not in config.EXCLUDE_IDS]
    print(len(docs))

    for group, titles in groups_map.items():
        group_docs = [doc for doc in docs if doc["title"] in titles]
        print(group)
        print(" ", len(group_docs), "docs")
        sample = random.sample(group_docs, config.SAMPLE_SIZE)
        si = similarity_index_symmetric(dbr, sample)
        print(" ", si, "similarity index")
        for group2, titles2 in groups_map.items():
            group2_docs = [doc for doc in docs if doc["title"] in titles2]
            sample2 = random.sample(group2_docs, config.SAMPLE_SIZE)
            si = similarity_index(dbr, sample, sample2)
            print("  ", group2, si, "similarity index")


def similarity_index_symmetric(dbr: db.DBReader, sample: list[db.DBDoc]):
    vectors = [dbr.get_context_vectors(doc) for doc in sample]
    similarity_matrix = cosine_similarity(vectors, vectors)
    # because X = Y, remove diagonal and symmetric elements
    # keep only one of M(i,j), M(j,i), remove M(i,i)
    similarity_matrix = np.triu(similarity_matrix, k=1)
    elements = similarity_matrix[np.nonzero(similarity_matrix)]
    mean = elements.mean()
    return mean


def similarity_index(
    dbr: db.DBReader, sample1: list[db.DBDoc], sample2: list[db.DBDoc]
):
    vectors1 = [dbr.get_context_vectors(doc) for doc in sample1]
    vectors2 = [dbr.get_context_vectors(doc) for doc in sample2]
    similarity_matrix = cosine_similarity(vectors1, vectors2)
    mean = similarity_matrix.mean()
    return mean


if __name__ == "__main__":
    main()
