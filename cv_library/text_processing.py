import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from collections.abc import Iterable
import fire
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import string

from database_utils import (
    CONNECTION_STRING, DATABASE_NAME,
    connect_to_database, get_contents_indices, get_documents_from_mongo, read_data
)

def _tokenize(text: str) -> list[str]:
    stemmer = PorterStemmer()
    translate_table = dict((ord(char), None) for char in string.punctuation)

    text = text.translate(translate_table)
    text = text.lower()

    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    words = [stemmer.stem(w) for w in words]

    return words

def get_tfidf(
    strings: Iterable[list[str]]
) -> list[list[float]]:
    tokens = map(lambda s: " ".join(_tokenize(s)), strings)

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(tokens)
    return vectors.toarray()

def get_all_tfidf(
    contents_index_file: str,
    contents_data_file: str,
    out_file: str,
    connection_string: str = CONNECTION_STRING,
    database_name: str = DATABASE_NAME
):
    # Get document ids from Mongo and indices for contents data file
    contents_indices = get_contents_indices(contents_index_file)
    database = connect_to_database(connection_string, database_name)

    # Get all strings
    texts_map = {}
    for collection_name, document_id in get_documents_from_mongo(database):
        doc = database[collection_name].find_one({"_id": document_id})
        title = doc["title"]
        header_name = doc["header_name"]
        if title in contents_indices:
            index = contents_indices[title]
            try:
                article = read_data(contents_data_file, index)
                text = article['text']
                texts_map[document_id] = text
            except:
                pass
        print(f"Processed {title}\\{header_name}")

    # Do TFIDF vectorization
    tfidfs = get_tfidf(texts_map.values())
    document_tfidfs = {d: t for d, t in zip(texts_map.keys(), tfidfs)}

    # Write to designated file
    with open(out_file, 'w') as file:
        json.dump(document_tfidfs, file)

if __name__ == "__main__":
    fire.Fire(get_all_tfidf)