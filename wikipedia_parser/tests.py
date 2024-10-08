import tempfile
from . import IndexedFlatFile


with tempfile.TemporaryFile("rt+") as index_file:
    with tempfile.TemporaryFile("rt+") as data_file:
        index_file.writelines(
            [
                "0:Apple\n",
                "10:Boy\n",
                "20:Cat\n",
            ]
        )
        index_file.flush()
        index_file.seek(0)
        data_file.write("0123456789ABCDEFGHIJQWERT")
        data_file.flush()
        data_file.seek(0)

        indexed_file = IndexedFlatFile.open(index_file, data_file)

        value = indexed_file.get("Apple")
        assert value == "0123456789", value
        value = indexed_file.get("Boy")
        assert value == "ABCDEFGHIJ", value
        value = indexed_file.get("Cat")
        assert value == "QWERT", value

print()
print("OK")
