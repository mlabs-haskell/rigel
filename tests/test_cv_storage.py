from cv_storage import ContextVectorDB

import tempfile
from pathlib import Path

import torch

def test_append_and_read_all():
    with tempfile.TemporaryDirectory() as tempdir:
        db = ContextVectorDB(Path(tempdir))

        db.insert("title1", "section1", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))
        db.insert("title2", "section1", torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))
        db.insert("title2", "section2", torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

        array = db.get("title1", "section1")
        assert array is not None
        assert torch.allclose(array, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))
        assert array.dtype == torch.float16

        array = db.get("title2", "section1")
        assert array is not None
        assert torch.allclose(array, torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))
        assert array.dtype == torch.float32

        array = db.get("title2", "section2")
        assert array is not None
        assert torch.allclose(array, torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))
        assert array.dtype == torch.float64

        assert db.has_article("title1")
        assert db.has_article("title2")
        assert not db.has_article("title3")

        assert db.has_section("title1", "section1")
        assert db.has_section("title2", "section1")
        assert db.has_section("title2", "section2")

        assert db.get_article_titles() == ["title1", "title2"]
        assert db.get_section_names("title1") == ["section1"]

        assert db.get_section_names("title2") == ["section1", "section2"]


def test_append_close_open_and_read():
    with tempfile.TemporaryDirectory() as tempdir:
        db = ContextVectorDB(Path(tempdir))

        db.insert("title1", "section1", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))
        db.insert("title2", "section1", torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))
        db.insert("title2", "section2", torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

        del db

        db = ContextVectorDB(Path(tempdir))

        array = db.get("title1", "section1")
        assert array is not None
        assert torch.allclose(array, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))
        assert array.dtype == torch.float16

        array = db.get("title2", "section1")
        assert array is not None
        assert torch.allclose(array, torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))
        assert array.dtype == torch.float32

        array = db.get("title2", "section2")
        assert array is not None
        assert torch.allclose(array, torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))
        assert array.dtype == torch.float64

        assert db.has_article("title1")
        assert db.has_article("title2")
        assert not (db.has_article("title3"))

        assert db.has_section("title1", "section1")
        assert db.has_section("title2", "section1")
        assert db.has_section("title2", "section2")

        assert db.get_article_titles() == ["title1", "title2"]

        assert db.get_section_names("title1") == ["section1"]

        assert db.get_section_names("title2") == ["section1", "section2"]


def test_append_close_open_read_append():
    with tempfile.TemporaryDirectory() as tempdir:
        db = ContextVectorDB(Path(tempdir))

        db.insert("title1", "section1", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))

        del db
        db = ContextVectorDB(Path(tempdir))

        db.insert("title1", "section1", torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32))
        db.insert("title1", "section2", torch.tensor([2.0, 3.0, 4.0], dtype=torch.float16))

        del db
        db = ContextVectorDB(Path(tempdir))

        array = db.get("title1", "section1")
        assert array is not None
        assert torch.allclose(array, torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32))
        assert array.dtype == torch.float32

        array = db.get("title1", "section2")
        assert array is not None
        assert torch.allclose(array, torch.tensor([2.0, 3.0, 4.0], dtype=torch.float16))
        assert array.dtype == torch.float16
