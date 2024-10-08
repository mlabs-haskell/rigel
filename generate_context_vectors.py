from collections.abc import Iterator
import json
import time

import fire
from tqdm import tqdm

from cv_storage import IndexedContextVectorDB
from modified_llama.llama import Llama
import wikipedia_parser

# A generating function that produces articles that haven't been processed yet
def document_iterator(
    cv_db: IndexedContextVectorDB,
    article_list: list[str],
    article_database: wikipedia_parser.IndexedFlatFile,
) -> Iterator[dict]:
    for article_title in article_list:
        # Only yield the article if it's not in the database yet
        if cv_db.get(article_title) is None:
            article_json = article_database.get(article_title)
            if "}{" in article_json:
                article_json, _ = article_json.split("}{")
                article_json += "}"

            # Read as a JSON
            article = json.loads(article_json)
            yield article

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

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    content_data_file: str,
    content_index_file: str,
    cv_index_file: str,
    cv_data_file: str,
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

    cv_db = IndexedContextVectorDB.open(open(cv_index_file), open(cv_data_file))
    articles_db = wikipedia_parser.IndexedFlatFile.open(open(content_index_file), open(content_data_file))

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
    for article in document_iterator(cv_db, article_list, articles_db):
        start = time.time()
        print("Processing", article["section_name"])

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
                [toks for _, toks in batched_tokens],
                max_gen_len
            )

            for j in range(len(batch_context_vectors)):
                (section, _) = batched_tokens[j]
                context_vectors.append((section, batch_context_vectors[j]))

        for section, context_vector in context_vectors:
            # TODO: section or article["section_name"]
            # @nate please clarify
            cv_db.insert(section, context_vector)

        elapsed = time.time() - start
        print(f"Finished processing {article['section_name']} in {elapsed} seconds")

if __name__ == "__main__":
    fire.Fire(main)
