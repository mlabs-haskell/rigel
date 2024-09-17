from collections.abc import Iterator
import json
import os
from pathlib import Path
import pickle
import re
import time

import fire
from tqdm import tqdm

from modified_llama.llama import Llama

def generate_texts(article) -> Iterator[tuple[str, str]]:
    # Generate text per section
    def get_text_by_section(section_name, article):
        if section_name == "":
            section_name = "root"
        else:
            subsection_name = article["section_name"]
            section_name = f"{section_name}\{subsection_name}"

        yield (section_name, article["text"])

        for child in article["children"]:
            yield from get_text_by_section(section_name, child)
    yield from get_text_by_section("", article)

# A generating function that produces articles that haven't been processed yet
def document_iterator(
    cv_dir_root: Path,
    article_list: list[str],
    content_data_file: str,
    offsets: dict[str, tuple[int, int]]
) -> Iterator[dict]:
    regex = re.compile(r"[^0-9a-zA-Z]")
    for article_title in article_list:
        # Only yield the article if it's not been processed yet
        article_path = cv_dir_root / (regex.sub("_", article_title) + ".pkl")
        if not article_path.is_file():
            with open(content_data_file, "r") as file:
                # Get the article from the contents monofile
                offset, length = offsets[article_title]
                file.seek(offset)
                article_json = file.read(length)
                if "\n}{" in article_json:
                    article_json, *_ = article_json.split("\n}{")
                    article_json += "\n}"

                # Read as a JSON
                try:
                    article = json.loads(article_json)
                except:
                    print("=============DUMP===========")
                    print(article_json)
                    print("=============END============")
                    exit(1)
                yield article
        else:
            print(f"Skipping {article_title}; already processed")

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    content_data_file: str,
    content_index_file: str,
    article_list_file: str,
    cv_dir_root: str,
    max_seq_len: int = 256,
    max_gen_len: int = 0,
    max_batch_size: int = 1,
):
    # Read the list of articles
    article_list = []
    with open(article_list_file, "r") as file:
        for line in file:
            article_list.append(line.strip())

    # Get offsets
    curr_offset = None
    curr_title = None
    offsets: dict[str, tuple[int, int]] = {}
    with open(content_index_file, "r") as file:
        for line in file:
            offset, title = line.strip().split(": ", 1)
            offset = int(offset)
            if curr_offset is not None:
                length = offset - curr_offset
                offsets[curr_title] = (curr_offset, length)
            curr_offset = offset
            curr_title = title

    # Get final offset
    content_data_file_stats = os.stat(content_data_file)
    length = content_data_file_stats.st_size - curr_offset
    offsets[curr_title] = (curr_offset, length)

    # Create the generator - very resource intensive
    print("Building generator")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("Built generator")

    # Iterate through each unprocessed article, get its context vectors, and write to the db
    regex = re.compile(r"[^0-9a-zA-Z]")
    cv_dir_root = Path(cv_dir_root)
    for article in document_iterator(cv_dir_root, article_list, content_data_file, offsets):
        start = time.time()
        article_title = article["section_name"]
        print("Processing", article_title)
        article_path = cv_dir_root / (regex.sub("_", article_title) + ".pkl")

        # Get the article texts and tokenize them
        texts = list(generate_texts(article))
        tokens = generator.tokenize(max_seq_len, texts)

        # Generate the context vectors for the documents
        context_vectors = {}
        pbar = tqdm(range(0, len(tokens), max_batch_size))
        for i in pbar:
            # Batch the tokenized texts
            batched_tokens = tokens[i : i + max_batch_size]
            batch_context_vectors = generator.generate(
                [toks for _, toks in batched_tokens],
                max_gen_len
            )

            # Get a tensor from the dict
            for _, tensor in batch_context_vectors.items():
                for j in range(len(tensor)):
                    (section, _) = batched_tokens[j]
                    context_vectors[section] = tensor[j]

        # Construct the pickle document
        document = {
            "title": article_title,
            "section_cv_map": context_vectors
        }
        with open(article_path, 'wb') as file:
            pickle.dump(document, file)

        elapsed = time.time() - start
        print(f"Finished processing {article['section_name']} in {elapsed} seconds")

if __name__ == "__main__":
    fire.Fire(main)
