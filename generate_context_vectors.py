from collections.abc import Iterator
import json
import time

import fire
import torch
from tqdm import tqdm

from cv_storage import ContextVectorDB, IndexedContextVectorDB
from modified_llama.llama import Llama
from wikipedia_parser import IndexedFlatFile
from wikipedia_parser.articles import generate_texts


# A generating function that produces articles that haven't been processed yet
def document_iterator(
    cv_db: ContextVectorDB,
    article_list: list[str],
    article_database: IndexedFlatFile,
) -> Iterator[dict]:
    for article_title in article_list:
        # Only yield the article if it's not in the database yet
        if cv_db.has_article(article_title):
            continue
        article_json = article_database.get(article_title)
        # Create a dictionary from JSON string
        article = json.loads(article_json)
        yield article


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    content_data_file: str,
    content_index_file: str,
    cv_db_folder: str,
    article_list_file: str,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    # Read the list of articles
    article_list = []
    with open(article_list_file, "r") as file:
        for line in file:
            article_list.append(line.strip())

    # Create the generator - very resource intensive
    print("Building generator")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("Built generator")

    # Create the context vector database
    cv_db = IndexedContextVectorDB(cv_db_folder)
    articles_db = IndexedFlatFile(content_index_file, content_data_file)

    # Iterate through each unprocessed article, get its context vectors, and write to the db
    for article in document_iterator(cv_db, article_list, articles_db):
        start = time.time()
        article_title = article["section_name"]
        print("Processing", article_title)

        # Get the article texts and tokenize them
        texts = list(generate_texts(article))
        tokens = generator.tokenize(max_seq_len, texts)

        # Generate the context vectors for the documents
        context_vectors = []
        pbar = tqdm(range(0, len(tokens), max_batch_size))
        for i in pbar:
            # Batch the tokenized texts
            batched_tokens = tokens[i : i + max_batch_size]
            batch_context_vectors = generator.generate(
                [toks for _, toks in batched_tokens], max_gen_len
            )

            for j in range(len(batch_context_vectors)):
                (section_name, _) = batched_tokens[j]
                context_vectors.append((section_name, batch_context_vectors[j]))

        # Insert context vectors into DB
        for section_name, context_vector in context_vectors:
            cv_db.insert(
                article_title, section_name, context_vector.cpu().detach().numpy()
            )

        elapsed = time.time() - start
        print(f"Finished processing {article_title} in {elapsed} seconds")


if __name__ == "__main__":
    fire.Fire(main)
