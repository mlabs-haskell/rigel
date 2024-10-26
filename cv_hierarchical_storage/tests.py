from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cv_hierarchical_storage import Database, DBConfig


def test_basic_read_write():
    with TemporaryDirectory() as temp_dir:
        config = DBConfig(
            vec_max_size=4,
            vec_min_size=2,
            compression_factor=2,
            search_narrow_factor=2,
            compression_dimension=0,
        )

        def similarity_fn(a: np.ndarray, b: np.ndarray) -> float:
            return np.dot(
                a / np.linalg.norm(a),
                b / np.linalg.norm(b),
            )

        db = Database(Path(temp_dir), config, similarity_fn)

        vecs = [
            [np.array([1, 3]), np.array([1, 2, 3, 4])],
            [np.array([5, 3]), np.array([5, 4, 3, 2])],
            [np.array([4, 2]), np.array([4, 3, 2, 1])],
            [np.array([4, 1]), np.array([4, 2, 1, 3])],
            [np.array([7, 1]), np.array([7, 2, 1, 3])],
            [np.array([8, 1]), np.array([8, 6, 1, 3])],
            [np.array([4, 9]), np.array([4, 2, 9, 1])],
            [np.array([1, 3]), np.array([1, 1, 3, 3])],
            [np.array([40, 90]), np.array([40, 21, 92, 10])],
        ]

        for vec in vecs:
            db.insert(vec)

        results = db.search(
            [
                np.array([1.1, 3.1]),
                np.array([0.9, 1.9, 2.9, 3.9]),
            ]
        )

        assert np.allclose(results[0], np.array([1, 2, 3, 4])), results[0]
        assert np.allclose(results[1], np.array([1, 1, 3, 3])), results[1]


def main():
    test_basic_read_write()
    print("All tests passed!")


if __name__ == "__main__":
    main()
