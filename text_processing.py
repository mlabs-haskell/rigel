from cv_storage import ContextVectorDB
from wikipedia_parser import IndexedFlatFile
from wikipedia_parser.articles import extract_section_text

import nltk
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from collections.abc import Iterable
import fire
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import string


def _tokenize(text: str) -> list[str]:
    # Remove punctuation and lower case the text
    translate_table = dict((ord(char), None) for char in string.punctuation)
    text = text.translate(translate_table)
    text = text.lower()

    # Tokenize the text, removing stop words and stemming all others
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [stemmer.stem(w) for w in words]

    return words


def get_tfidf(strings: Iterable[list[str]]) -> list[list[float]]:
    tokens = map(lambda s: " ".join(_tokenize(s)), strings)

    tfidf = TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=1000)
    vectors = tfidf.fit_transform(tokens)
    return vectors.toarray().tolist()


def get_all_tfidf(
    contents_index_file: str,
    contents_data_file: str,
    out_file: str,
    cvdb_folder: str,
):
    article_contents_db = IndexedFlatFile(
        contents_index_file,
        contents_data_file,
    )
    cv_db = ContextVectorDB(cvdb_folder)

    texts_map = {}
    for article_title in cv_db.get_article_titles():
        article_str = article_contents_db.get(article_title)
        if article_str is None:
            print(f"Article {article_title} could not be found in the index file")
            continue

        article = json.loads(article_str)
        for section_name in cv_db.get_section_names(article_title):
            text = extract_section_text(section_name, article)
            texts_map[(article_title, section_name)] = text
            print(f"Processed {article_title}\\{section_name}")

    # Do TFIDF vectorization
    article_keys = list(texts_map.keys())
    tfidfs = get_tfidf(texts_map.values())

    document_tfidfs = {}
    for (article_title, section_name), tfidf in zip(article_keys, tfidfs):
        if article_title not in document_tfidfs:
            document_tfidfs[article_title] = {}

        document_tfidfs[article_title][section_name] = tfidf

    # Write to designated file
    with open(out_file, "w") as file:
        json.dump(document_tfidfs, file, indent=4)


if __name__ == "__main__":
    fire.Fire(get_all_tfidf)
