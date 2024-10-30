from cv_library.compressor import Compressor
from cv_library.loss_functions import sequence_similarity
from cv_storage import ContextVectorDB
import cv_hierarchical_storage as cvhs
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import shutil

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float16)

def get_all_headings(db: ContextVectorDB) -> list[tuple[str, str]]:
    res = []
    for article_title in db.get_article_titles():
        for section_name in db.get_section_names(article_title):
            res.append((article_title, section_name))
    return res


# Not used because need to figure out how to batch tensors of different sizes
# def make_batches(xs, batch_size):
#     for i in range(0, len(xs), batch_size):
#         yield xs[i:i+batch_size]
# 

def main():
    compressor_chkpt = "../rigel-data/hierarchical-compression-checkpoint-2024-10-30/attention_model.pt"
    compressor = Compressor(compressor_chkpt)
    #batch_size = 1024

    input_db_path = Path("../rigel-data/context-vectors-2024-10-30/")
    output_db_path = Path("../rigel-data/context-vectors-compressed-2024-10-30")

    # Currently we don't support resuming
    # To prevent inconsistent states, clean the DB before writing to it
    try:
        shutil.rmtree(output_db_path)
    except FileNotFoundError:
        pass
    output_db_path.mkdir()


    input_db = ContextVectorDB(input_db_path)
    output_db_config = cvhs.DBConfig(
            vec_max_size = 4096,
            vec_min_size = 8,
            compression_factor = 8,
            compression_dimension = 1, # cv.shape = (seq_size, 4096). We compress the one with 4096.
            search_narrow_factor = 16,
    )
    output_db = cvhs.Database(output_db_path, output_db_config, sequence_similarity)

    print("Reading metadata ..")
    headings = get_all_headings(input_db)
    # headings_batched = list(make_batches(headings, batch_size))
    
    print("Processing ..")
    # for batch in tqdm(headings_batched):
    #     cvs = []
    #     for article_title, section_name in batch:
    #         cvs.append(input_db.get(article_title, section_name))
    #     cvs = np.stack(cvs)
    #     cvs = torch.tensor(cvs)
    #     print("cvs", cvs.shape)
    #     output_cvs = compressor.compress(cvs)
    #     print("output_cvs", output_cvs.shape)

    for article_title, section_name in tqdm(headings):
        cv = input_db.get(article_title, section_name)
        cv = torch.tensor(cv).unsqueeze(dim=0)
        output_cvs = [cv, *compressor.compress(cv)]
        output_cvs_fixed = [cv.squeeze(dim=0).cpu().detach().numpy() for cv in output_cvs]
        output_cvs_fixed.reverse()
        output_db.insert(output_cvs_fixed)


if __name__ == "__main__":
    main()
