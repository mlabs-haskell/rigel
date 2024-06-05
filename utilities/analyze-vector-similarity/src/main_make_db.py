import dumps
import db
import config

files = range(1, 11)
dbw = db.DBWriter(config.DB_DIR)

for i in files:
    print("File:", i)
    file = dumps.open_dump_file_by_index(i)
    j = -1
    for j, doc in enumerate(file):
        print(j, end="\r")
        dbw.write_doc(doc)
    print("Done:", j + 1)
    print()
