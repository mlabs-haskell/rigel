import random
import numpy as np
import json
from tqdm import tqdm

import db
import config

SAMPLE_SIZE = 500


def cosine_similarity(a1: np.ndarray, a2: np.ndarray) -> float:
    return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))


def main():
    dbr = db.DBReader(config.DB_DIR)

    docs = dbr.read_docs()
    docs = [d for d in docs if d["id"] not in config.EXCLUDE_IDS]

    print("Loading context vectors ..")
    sample = random.sample(docs, SAMPLE_SIZE)
    cvs = [dbr.get_context_vectors(s) for s in sample]
    print("Done.")
    print()

    output_filename = config.OUT_DIR + "/full_similarity.json"
    output_file = open(output_filename, "w")

    for s1, cv1 in tqdm(zip(sample, cvs)):
        for s2, cv2 in zip(sample, cvs):
            si = cosine_similarity(cv1, cv2)
            doc = {
                "x": {
                    "id": s1["id"],
                    "title": s1["title"],
                    "header": s1["header_name"],
                },
                "y": {
                    "id": s2["id"],
                    "title": s2["title"],
                    "header": s2["header_name"],
                },
                "similarity_index": si,
            }
            output_file.write(json.dumps(doc))
            output_file.write("\n")


if __name__ == "__main__":
    main()
