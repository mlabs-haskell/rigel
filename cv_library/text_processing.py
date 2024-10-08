import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from collections.abc import Iterable
import fire
import json
import os
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import string

from utils import get_contents_indices, read_data

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

    tfidf = TfidfVectorizer(max_df=0.95, min_df=0.01)
    vectors = tfidf.fit_transform(tokens)
    return vectors.toarray().tolist()

def get_all_tfidf(
    contents_index_file: str,
    contents_data_file: str,
    out_file: str,
    cv_dir: str
):
    cv_dir_path = Path(cv_dir)

    # Get indices for contents data file
    contents_indices = get_contents_indices(contents_index_file)

    # Get all strings
    texts_map = {}
    delete_files = []
    for filename in os.listdir(cv_dir_path):
        # Open the file
        filepath = cv_dir_path / filename
        with open(filepath, 'rb') as file:
            try:
                doc = pickle.load(file)
            except:
                print(f"Skipping {filename}; file in incorrect format")
                delete_files.append(filepath)

            # Get article title and section header
            title = doc["title"]
            if title not in contents_indices:
                print(f"{title} not in contents file")
                continue
            index = contents_indices[title]

            # Read the article
            try:
                article = read_data(contents_data_file, index)
            except:
                print(f"Could not parse {title}")
                continue

            for header_name, cv in doc["section_cv_map"].items():
                # For the root, the text is the article's text field
                if header_name == 'root':
                    text = article['text']

                # Otherwise, look for the correct child
                else:
                    headers = header_name.split('\\')
                    headers = headers[1:]
                    section = article

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
                texts_map[f"{title}\\{header_name}"] = text
                print(f"Processed {title}\\{header_name}")

    # Remove bad files
    for delete_file in delete_files:
        os.remove(delete_file)

    # Do TFIDF vectorization
    texts = texts_map.values()
    tfidfs = get_tfidf(texts)
    document_tfidfs = {
        d: t
        for d, t in zip(texts_map.keys(), tfidfs)
        if any([v > 0.0 for v in t])
    }

    # Write to designated file
    with open(out_file, 'w') as file:
        json.dump(document_tfidfs, file, indent=4)

if __name__ == "__main__":
    fire.Fire(get_all_tfidf)
