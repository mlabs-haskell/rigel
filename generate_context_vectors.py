from collections.abc import Iterator
import json
import os

import fire
from pymongo import MongoClient
from pymongo.collection import Collection

from modified_llama.llama import Llama

CONNECTION_STRING = "mongodb://127.0.0.1"
DATABASE_NAME = "rigel"
COLLECTION_NAME = "context_vectors"
ARTICLES_DIR = "parsed_articles"

"""
schema:
{
    "title": str,
    "context_vectors": [{
        "scope": "full_without_headers" | "full_with_headers" | "sections",
        "header_name"?: str,          # only if scope == sections
        "text_offset": int,
        "context_vectors": binData
    }]
}
"""

# Load the MongoDB collection we are saving the context vectors to
def get_collection():
    client = MongoClient(CONNECTION_STRING)
    return client[DATABASE_NAME][COLLECTION_NAME]

# A generating function that produces articles that haven't been processed yet
def document_iterator(collection: Collection) -> Iterator[dict]:
    for dirpath, _, files in os.walk(ARTICLES_DIR):
        for file in files:
            with open(os.path.join(dirpath, file), "r") as f:
                # Interpret the article's JSON
                text = f.read()
                article = json.loads(text)
                
                # Only yield the article if it's not in the database yet
                if collection.find_one({"title": article["section_name"]}) is None:
                    yield article
                    
def generate_texts(article) -> Iterator[tuple[str, str]]:
    # Use a recursive function to fetch text from all child sections of the article
    def flatten_article_tree(article):
        texts = [(article["section_name"], article["text"])]
        for child in article["children"]:
            texts += flatten_article_tree(child)
        return texts
    all_section_texts = flatten_article_tree(article)
    
    # Generate full text without headers
    text = "\n".join([t for _, t in all_section_texts])
    yield ("full_without_headers", text)
    
    # Generate full text with headers
    text_and_headers = []
    for section_name, section_text in all_section_texts:
        text_and_headers.append(section_name)
        text_and_headers.append(section_text)
    yield ("full_with_headers", "\n".join(text_and_headers))
    
    # Generate text per section
    def get_text_by_section(section_name, article):
        if section_name == "":
            yield ("root", article["text"])
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
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
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
    collection = get_collection()
    for article in document_iterator(collection):
        # Get the article texts and tokenize them
        texts = list(generate_texts(article))
        tokens = generator.tokenize(max_seq_len, texts)
        
        # Generate the context vectors for the documents
        context_vectors = []
        for i in range(0, len(tokens), max_batch_size):
            # Batch the tokenized texts
            batched_tokens = tokens[i : i + max_batch_size]
            print(batched_tokens)
            batch_context_vectors = generator.generate(
                [toks for _, _, toks in batched_tokens],
                max_gen_len
            )
            
            for j in range(len(batch_context_vectors)):
                (scope, offset, _) = batched_tokens[j]
                context_vectors.append((scope, offset, batch_context_vectors[j]))
            
        # Construct the "context_vectors" field of the MongoDB document
        db_output = []
        for scope, offset, cvs in context_vectors:
            document = {
                "text_offset": offset,
                "context_vectors": cvs
            }
            
            if scope == "full_without_headers" or scope == "full_with_headers":
                document["scope"] = scope
            else:
                document["scope"] = "sections"
                document["header_name"] = scope
                
            db_output.append(document)
           
        # Construct the full, final MongoDB document
        final_document = {
            "title": article["section_name"],
            "context_vectors": db_output
        }
        collection.insert_one(final_document)

if __name__ == "__main__":
    fire.Fire(main)