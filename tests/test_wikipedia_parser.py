from pathlib import Path
import tempfile

from ..wikipedia_parser import IndexedFlatFile


def test_wikipedia_parser():
    with tempfile.TemporaryDirectory() as temp_dir:
        p = Path(temp_dir)
        index_file = p / "index.txt"
        data_file = p / "data.txt"

        with open(index_file, "wt") as f:
            f.write("0: Apple\n")
            f.write("10: Boy\n")
            f.write("20: Cat\n")

        with open(data_file, "wt") as f:
            f.write("0123456789ABCDEFGHIJQWERT")

        indexed_file = IndexedFlatFile(str(index_file), str(data_file))

        value = indexed_file.get("Apple")
        assert value == "0123456789", value
        value = indexed_file.get("Boy")
        assert value == "ABCDEFGHIJ", value
        value = indexed_file.get("Cat")
        assert value == "QWERT", value
