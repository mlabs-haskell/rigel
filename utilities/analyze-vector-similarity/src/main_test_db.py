import dumps
import db
import numpy as np

files = range(1, 11)
dbr = db.DBReader("db")

db_docs = dbr.read_docs()

for i in files:
    print("File:", i)
    file = dumps.open_dump_file_by_index(i)
    j = -1
    for j, doc in enumerate(file):
        len(doc["context_vector"])
        print(j, end="\r")
        db_doc = next(filter(lambda d: d["id"] == str(doc["_id"]), db_docs))
        db_cv = dbr.get_context_vectors(db_doc)
        cv = np.array(doc["context_vector"]).flatten()
        assert (cv == db_cv).all()

    print("Done:", j + 1)
    print()
