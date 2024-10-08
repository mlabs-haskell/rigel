import numpy as np
from . import ContextVectorDB

# Generate tests for ContextVectorDB class using unittest module
import unittest
import tempfile
from pathlib import Path


class TestContextVectorDB(unittest.TestCase):

    def test_append_and_read_all(self):
        with tempfile.TemporaryDirectory() as tempdir:
            main_file = open(Path(tempdir) / "temp.main.cvdb", "a+b")
            metadata_file = open(Path(tempdir) / "temp.meta.cvdb", "a+b")

            db = ContextVectorDB(main_file, metadata_file)

            db.append_context_vector("title1", "section1", np.array([1.0, 2.0, 3.0]))
            db.append_context_vector("title2", "section1", np.array([4.0, 5.0, 6.0]))
            db.append_context_vector("title2", "section2", np.array([7.0, 8.0, 9.0]))

            iter = db.read_context_vectors()

            title, section, array = next(iter)
            self.assertEqual(title, "title1")
            self.assertEqual(section, "section1")
            self.assertTrue(np.allclose(array, np.array([1.0, 2.0, 3.0])))

            title, section, array = next(iter)
            self.assertEqual(title, "title2")
            self.assertEqual(section, "section1")
            self.assertTrue(np.allclose(array, np.array([4.0, 5.0, 6.0])))

            title, section, array = next(iter)
            self.assertEqual(title, "title2")
            self.assertEqual(section, "section2")
            self.assertTrue(np.allclose(array, np.array([7.0, 8.0, 9.0])))

            with self.assertRaises(StopIteration):
                title, array = next(iter)

            main_file.close()
            metadata_file.close()

    def test_append_close_open_and_read(self):
        with tempfile.TemporaryDirectory() as tempdir:
            main_file = open(Path(tempdir) / "temp.main.cvdb", "wb")
            metadata_file = open(Path(tempdir) / "temp.meta.cvdb", "wb")

            db = ContextVectorDB(main_file, metadata_file)

            db.append_context_vector("title1", "section1", np.array([1.0, 2.0, 3.0]))
            db.append_context_vector("title2", "section1", np.array([4.0, 5.0, 6.0]))
            db.append_context_vector("title2", "section2", np.array([7.0, 8.0, 9.0]))

            main_file.close()
            metadata_file.close()

            main_file = open(Path(tempdir) / "temp.main.cvdb", "rb")
            metadata_file = open(Path(tempdir) / "temp.meta.cvdb", "rb")

            db = ContextVectorDB(main_file, metadata_file)

            iter = db.read_context_vectors()

            title, section, array = next(iter)
            self.assertEqual(title, "title1")
            self.assertEqual(section, "section1")
            self.assertTrue(np.allclose(array, np.array([1.0, 2.0, 3.0])))

            title, section, array = next(iter)
            self.assertEqual(title, "title2")
            self.assertEqual(section, "section1")
            self.assertTrue(np.allclose(array, np.array([4.0, 5.0, 6.0])))

            title, section, array = next(iter)
            self.assertEqual(title, "title2")
            self.assertEqual(section, "section2")
            self.assertTrue(np.allclose(array, np.array([7.0, 8.0, 9.0])))

            with self.assertRaises(StopIteration):
                title, array = next(iter)

            main_file.close()
            metadata_file.close()


if __name__ == "__main__":
    unittest.main()
