from collections.abc import Iterator
import hashlib
import json
import os
import time

import fire
from tqdm import tqdm
from pymongo import MongoClient
from pymongo.collection import Collection

from modified_llama.llama import Llama

CONNECTION_STRING = "mongodb://127.0.0.1"
DATABASE_NAME = "rigel"

"""
schema:
{
    "title": str,
    "header_name": str
    "context_vector": binData
}
"""

# Load the MongoDB collection we are saving the context vectors to
def get_collection(article_name: str):
    # Put the article into an effectively random collection
    # This is done to improve Mongo performance
    checksum = hashlib.md5(article_name.encode('utf-8')).hexdigest() 
    collection_number = int(checksum, 16) % 100
    collection_name = f"bin_{collection_number}"
    
    client = MongoClient(CONNECTION_STRING)
    return client[DATABASE_NAME][collection_name]

# A generating function that produces articles that haven't been processed yet
def document_iterator(
    article_list: list[str],
    content_data_file: str,
    offsets: dict[str, tuple[int, int]]
) -> Iterator[dict]:
    for article_title in article_list:
        # Connect to collection
        collection = get_collection(article_title)
        
        # Only yield the article if it's not in the database yet
        if collection.find_one({"title": article_title}) is None:
            with open(content_data_file, "r") as file:
                # Get the article from the contents monofile
                offset, length = offsets[article_title]
                file.seek(offset)
                article_json = file.read(length)
                if "}{" in article_json:
                    article_json, *_ = article_json.split("}{")
                    article_json += "}"

                # Read as a JSON
                article = json.loads(article_json)
                yield collection, article
        
        else:
            print(f"Skipping {article_title}")
                    
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
    for collection, article in document_iterator(article_list, content_data_file, offsets):
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

            # This will return a list of size (number_of_layers, number_of_batches)
            # ie, list[i] will give us the context vectors at layer i.
            batch_context_vectors_list = generator.generate(
                [toks for _, toks in batched_tokens],
                max_gen_len
            )
            
            for j in range(len(batched_tokens)):
                (section, _) = batched_tokens[j]
                # Get j'th batch for each layer
                batch_context_vectors = [layer[j] for layer in batch_context_vectors_list]
                context_vectors.append((section, batch_context_vectors))
            
        # Construct the MongoDB document
        for section, context_vectors_list in context_vectors:
            # MongoDB fails to insert all context vectors into a single document due to size limits.
            # Avg document size is 15x larger than the max allowed size.
            # So I'm spliting the list of vectors into chunks and inserting them as multiple documents.
            # I'm using 20 as the chunk size to allow some head room for large title or header_name values.
            max_context_vector_list_size = len(context_vectors_list) // 20

            documents = []
            for i in range(0, len(context_vectors_list), max_context_vector_list_size):
                context_vectors_list_chunk = context_vectors_list[i:i+max_context_vector_list_size]
                document = {
                    "title": article["section_name"],
                    "header_name": section,
                    "context_vectors_list": context_vectors_list_chunk,
                }
                documents.append(document)
            collection.insert_many(documents)
        
        elapsed = time.time() - start
        print(f"Finished processing {article['section_name']} in {elapsed} seconds")

if __name__ == "__main__":
    fire.Fire(main)
    
