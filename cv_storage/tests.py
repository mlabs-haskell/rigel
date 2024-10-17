import numpy as np
from . import ContextVectorDB

# Generate tests for ContextVectorDB class using unittest module
import unittest
import tempfile
from pathlib import Path


class TestContextVectorDB(unittest.TestCase):

    def test_append_and_read_all(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db = ContextVectorDB(Path(tempdir))

            db.insert("title1", "section1", np.array([1.0, 2.0, 3.0], dtype="float16"))
            db.insert("title2", "section1", np.array([4.0, 5.0, 6.0], dtype="float32"))
            db.insert("title2", "section2", np.array([7.0, 8.0, 9.0], dtype="float64"))

            array = db.get("title1", "section1")
            self.assertTrue(np.allclose(array, np.array([1.0, 2.0, 3.0])))
            self.assertEqual(array.dtype, np.float16)

            array = db.get("title2", "section1")
            self.assertTrue(np.allclose(array, np.array([4.0, 5.0, 6.0])))
            self.assertEqual(array.dtype, np.float32)

            array = db.get("title2", "section2")
            self.assertTrue(np.allclose(array, np.array([7.0, 8.0, 9.0])))
            self.assertEqual(array.dtype, np.float64)

            self.assertTrue(db.has_article("title1"))
            self.assertTrue(db.has_article("title2"))
            self.assertFalse(db.has_article("title3"))

            self.assertTrue(db.has_section("title1", "section1"))
            self.assertTrue(db.has_section("title2", "section1"))
            self.assertTrue(db.has_section("title2", "section2"))

            self.assertEqual(
                db.get_article_titles(),
                ["title1", "title2"],
            )

            self.assertEqual(db.get_section_names("title1"), ["section1"])

            self.assertEqual(
                db.get_section_names("title2"),
                ["section1", "section2"],
            )

    def test_append_close_open_and_read(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db = ContextVectorDB(Path(tempdir))

            db.insert("title1", "section1", np.array([1.0, 2.0, 3.0], dtype="float16"))
            db.insert("title2", "section1", np.array([4.0, 5.0, 6.0], dtype="float32"))
            db.insert("title2", "section2", np.array([7.0, 8.0, 9.0], dtype="float64"))

            del db

            db = ContextVectorDB(Path(tempdir))

            array = db.get("title1", "section1")
            self.assertTrue(np.allclose(array, np.array([1.0, 2.0, 3.0])))
            self.assertEqual(array.dtype, np.float16)

            array = db.get("title2", "section1")
            self.assertTrue(np.allclose(array, np.array([4.0, 5.0, 6.0])))
            self.assertEqual(array.dtype, np.float32)

            array = db.get("title2", "section2")
            self.assertTrue(np.allclose(array, np.array([7.0, 8.0, 9.0])))
            self.assertEqual(array.dtype, np.float64)

            self.assertTrue(db.has_article("title1"))
            self.assertTrue(db.has_article("title2"))
            self.assertFalse(db.has_article("title3"))

            self.assertTrue(db.has_section("title1", "section1"))
            self.assertTrue(db.has_section("title2", "section1"))
            self.assertTrue(db.has_section("title2", "section2"))

            self.assertEqual(
                db.get_article_titles(),
                ["title1", "title2"],
            )

            self.assertEqual(db.get_section_names("title1"), ["section1"])

            self.assertEqual(
                db.get_section_names("title2"),
                ["section1", "section2"],
            )

    def test_append_close_open_read_append(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db = ContextVectorDB(Path(tempdir))

            db.insert("title1", "section1", np.array([1.0, 2.0, 3.0], dtype="float16"))

            del db
            db = ContextVectorDB(Path(tempdir))

            db.insert("title1", "section1", np.array([1.0, 2.0, 4.0], dtype="float32"))
            db.insert("title1", "section2", np.array([2.0, 3.0, 4.0], dtype="float16"))

            del db
            db = ContextVectorDB(Path(tempdir))

            array = db.get("title1", "section1")
            self.assertTrue(np.allclose(array, np.array([1.0, 2.0, 4.0])))
            self.assertEqual(array.dtype, np.float32)

            array = db.get("title1", "section2")
            self.assertTrue(np.allclose(array, np.array([2.0, 3.0, 4.0])))
            self.assertEqual(array.dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
