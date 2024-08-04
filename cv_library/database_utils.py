from collections.abc import Iterator
import json
from pymongo import MongoClient
from pymongo.database import Database

CONNECTION_STRING = "mongodb://127.0.0.1"
DATABASE_NAME = "rigel"

def connect_to_database(
    connection_string: str = CONNECTION_STRING,
    database_name: str = DATABASE_NAME
) -> Database:
    client = MongoClient(connection_string)
    database = client[database_name]
    return database

def get_contents_indices(contents_index_file: str) -> dict[str, int]:
    contents_indices = {}
    with open(contents_index_file, 'r') as file:
        for line in file:
            line = line.strip()
            [index, article_name] = line.split(': ', maxsplit=1)
            contents_indices[article_name] = int(index)
    return contents_indices

def get_documents_from_mongo(
    database: Database
) -> Iterator:
    # Get all document ids
    for collection_name in database.list_collection_names():
        collection = database[collection_name]
        for document in collection.find():
            yield document

# Reads JSON object from data file starting at given index
def read_data(contents_data_file: str, index: int) -> dict:
    with open(contents_data_file, 'r') as file:
        # Go to index in file
        file.seek(index)

        # Read JSON string one char at a time, keeping track of open braces
        buffer = ""
        num_open_braces = 0
        while True:
            c = file.read(1)
            if c == '{':
                num_open_braces += 1
            if c == '}':
                num_open_braces -= 1
            buffer += c

            if num_open_braces == 0:
                break

    ret_val = json.loads(buffer)
    return ret_val