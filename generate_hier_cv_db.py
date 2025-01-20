import time
from typing import Sequence

from cv_library.compressor import Compressor
from cv_library.loss_functions import sequence_similarity
from cv_storage import ContextVectorDB
import cv_storage.cv_hier_storage as cvhs

from fire import Fire
from tqdm import tqdm
import torch

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timer(description="Execution time"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{description}: {elapsed:.4f} seconds")


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)


def get_all_headings(db: ContextVectorDB) -> list[tuple[str, str]]:
    res = []
    for article_title in db.get_article_titles():
        for section_name in db.get_section_names(article_title):
            res.append((article_title, section_name))
    return res

@torch.no_grad()
def main(
    compressor_chkpt: str,
    cv_db_dir: str,
    hier_db_dir: str,
    level_sizes: Sequence[int],
    mode: str = "generate",
    limit: int | None = None,
    search_narrow_factor: int | None = None,
    max_level: int | None = None,
):
    for x in level_sizes:
        assert x > 0

    compressor = Compressor(compressor_chkpt)
    print("Loaded compressor")

    cv_db_path = Path(cv_db_dir)
    hier_db_path = Path(hier_db_dir)

    if mode == "generate":
        if hier_db_path.exists():
            print("Output folder exists: " + str(hier_db_path))
            print("Remove it and run again to regenerate")
            exit(-1)

        hier_db_path.mkdir()

    cv_db = ContextVectorDB(cv_db_path)
    hier_db_config = cvhs.DBConfig(level_sizes=list(level_sizes))
    hier_db = cvhs.ContextVectorHierDB(
        hier_db_path, hier_db_config, sequence_similarity
    )

    print("Reading metadata ..")
    headings = get_all_headings(cv_db)
    if limit is not None:
        headings = headings[:limit]

    if mode == "generate":
        print("Generating ..")
        generate_db(cv_db, headings, compressor, hier_db)
    elif mode == "verify":
        assert search_narrow_factor is not None
        print("Verifying ..")
        verify_db(
            cv_db,
            headings,
            compressor,
            hier_db,
            search_narrow_factor,
            max_level,
        )
    else:
        print("Unknown mode:", mode)
        print("Available options: generate, verify")

def to_hierarchical(cv: torch.Tensor, compressor: Compressor) -> list[torch.Tensor]:
    cv = cv.to(dtype=DTYPE).unsqueeze(dim=0)

    hier_cvs = [cv, *compressor.compress(cv)]
    hier_cvs.reverse()
    hier_cvs = [tensor.squeeze(dim=0) for tensor in hier_cvs]

    return hier_cvs


def generate_db(
    cv_db: ContextVectorDB,
    headings: list[tuple[str, str]],
    compressor: Compressor,
    hier_db: cvhs.ContextVectorHierDB,
):
    for article_title, section_name in tqdm(headings):
        cv = cv_db.get(article_title, section_name)
        assert cv is not None
        hier_cvs = to_hierarchical(cv, compressor)

        metadata = cvhs.CVMetadata(article_title, section_name)
        hier_db.insert(metadata, hier_cvs)


def verify_db(
    cv_db: ContextVectorDB,
    headings: list[tuple[str, str]],
    compressor: Compressor,
    hier_db: cvhs.ContextVectorHierDB,
    search_narrow_factor: int,
    max_level: int | None,
):
    for i, (article_title, section_name) in enumerate(tqdm(headings)):
        cv = cv_db.get(article_title, section_name)
        assert cv is not None
        hier_cvs = to_hierarchical(cv, compressor)

        meta_ = hier_db.get_metadata(i)
        assert meta_.article_title == article_title
        assert meta_.section_name == section_name

        closest_cvs = hier_db.search(hier_cvs, search_narrow_factor, max_level)

        closest = closest_cvs[0].cv
        closest = closest.to(cv)
        assert torch.allclose(cv, closest)


if __name__ == "__main__":
    Fire(main)
