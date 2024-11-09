from pathlib import Path
from tempfile import TemporaryDirectory
from cv_library.loss_functions import sequence_similarity

import numpy as np

from cv_hier_storage import CVMetadata, ContextVectorHierDB, DBConfig


def test_basic_read_write():
    with TemporaryDirectory() as temp_dir:
        config = DBConfig(
            level_sizes=[2, 4],
        )

        db = ContextVectorHierDB(Path(temp_dir), config, sequence_similarity)

        arr = lambda xs: np.array([xs], dtype=np.float32)

        vecs = [
            [arr([1, 3]), arr([1, 2, 3, 4])],
            [arr([5, 3]), arr([5, 4, 3, 2])],
            [arr([4, 2]), arr([4, 3, 2, 1])],
            [arr([4, 1]), arr([4, 2, 1, 3])],
            [arr([7, 1]), arr([7, 2, 1, 3])],
            [arr([8, 1]), arr([8, 6, 1, 3])],
            [arr([4, 9]), arr([4, 2, 9, 1])],
            [arr([1, 3]), arr([1, 1, 3, 3])],
            [arr([40, 90]), arr([40, 21, 92, 10])],
        ]

        for i, vec in enumerate(vecs):
            db.insert(CVMetadata(str(i), "section"), vec)

        results = db.search(
            [
                arr([1.1, 3.1]),
                arr([0.9, 1.9, 2.9, 3.9]),
            ],
            2,
        )

        expected = [
            (CVMetadata("0", "section"), arr([1, 2, 3, 4])),
            (CVMetadata("7", "section"), arr([1, 1, 3, 3])),
        ]

        for res, (expected_meta, expected_cv) in zip(results, expected):
            meta = db.get_metadata(res.idx)
            assert meta == expected_meta, (meta, expected_meta, res)
            assert np.allclose(res.cv, expected_cv), res


def main():
    test_basic_read_write()
    print("All tests passed!")


if __name__ == "__main__":
    main()
