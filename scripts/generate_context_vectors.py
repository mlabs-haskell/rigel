import json
from pathlib import Path

import fire
from tqdm import tqdm

from cv_storage import ContextVectorDB
from modified_llama.llama import Llama
from wikipedia_parser import IndexedFlatFile
from wikipedia_parser.articles import generate_texts

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    content_data_file: str,
    content_index_file: str,
    cv_db_folder: str,
    article_list_file: str,
    transformer_position: int,
    max_seq_len: int = 128,
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
    cv_db_folder = Path(cv_db_folder)
    cv_db = ContextVectorDB(cv_db_folder)
    articles_db = IndexedFlatFile(content_index_file, content_data_file)

    # Iterate through each unprocessed article, get its context vectors, and write to the db
    for article_title in tqdm(article_list):
        # Only process the article if it's not in the database yet
        if cv_db.has_article(article_title):
            continue

        # Create a dictionary from JSON string
        article_json = articles_db.get(article_title)
        article = json.loads(article_json)

        # Get the article texts and tokenize them
        texts = list(generate_texts(article))
        tokens = generator.tokenize(max_seq_len, texts, False)

        # Generate the context vectors for the documents
        context_vectors = []
        for i in tqdm(range(0, len(tokens), max_batch_size), leave=False):
            # Batch the tokenized texts
            batched_tokens = tokens[i : i + max_batch_size]
            _, batch_context_vectors = generator.generate_context_vectors(
                [toks for _, toks in batched_tokens], 0
            )
            batch_context_vectors = batch_context_vectors[transformer_position]

            for j in range(len(batch_context_vectors)):
                (section_name, _) = batched_tokens[j]
                context_vectors.append((section_name, batch_context_vectors[j]))

        # Insert context vectors into DB
        article_title = article["section_name"]
        for section_name, context_vector in context_vectors:
            cv_db.insert(
                article_title, section_name, context_vector
            )

if __name__ == "__main__":
    fire.Fire(main)
