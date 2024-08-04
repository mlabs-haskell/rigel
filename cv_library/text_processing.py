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
    # Remove punctuation and lower case the text
    translate_table = dict((ord(char), None) for char in string.punctuation)
    text = text.translate(translate_table)
    text = text.lower()

    # Tokenize the text, removing stop words and stemming all others
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    words = [stemmer.stem(w) for w in words]

    return words

def get_tfidf(
    strings: Iterable[list[str]]
) -> list[list[float]]:
    tokens = map(lambda s: " ".join(_tokenize(s)), strings)

    tfidf = TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=1000)
    vectors = tfidf.fit_transform(tokens)
    return vectors.toarray().tolist()

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
    last_title = None
    last_article = None
    for doc in get_documents_from_mongo(database):
        # Get article title and section header
        document_id = doc['_id']
        title = doc["title"]
        header_name = doc["header_name"]

        # Only process titles that are in both the db and the index file
        if title in contents_indices:
            # Read the data from the contents file only if it hasn't been read
            if last_title != title:
                index = contents_indices[title]
                last_title = title
                try:
                    article = read_data(contents_data_file, index)
                    last_article = article
                except:
                    print(f"Could not parse {title}\\{header_name}")
                    last_article = None
                    continue

            # Skip articles whose JSON can't be parsed
            if last_article is None:
                continue

            # For the root, the text is the article's text field
            if header_name == 'root':
                text = last_article['text']

            # Otherwise, look for the correct child
            else:
                headers = header_name.split('\\')
                headers = headers[1:]
                section = last_article

                # Descend through the tree until we reach the proper section
                for header in headers:
                    found_section = False
                    for child in section['children']:
                        if child['section_name'] == header:
                            section = child
                            found_section = True
                            break

                    if not found_section:
                        print(f"Could not find section {title}\\{header_name}")
                        section = None
                        break

                # If the section could not be found, move on
                if section is None:
                    continue

                text = section['text']
            texts_map[str(document_id)] = text
            print(f"Processed {title}\\{header_name}")

        else:
            print(f"Article {title} could not be found in the index file")

    # Do TFIDF vectorization
    tfidfs = get_tfidf(texts_map.values())
    document_tfidfs = {d: t for d, t in zip(texts_map.keys(), tfidfs)}

    # Write to designated file
    with open(out_file, 'w') as file:
        json.dump(document_tfidfs, file, indent=4)

if __name__ == "__main__":
    fire.Fire(get_all_tfidf)