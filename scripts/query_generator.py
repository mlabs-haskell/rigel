import fire
import itertools
import json
import re
import sys
from wikipedia_parser import IndexedFlatFile

def generate_query(
    wiki_links: IndexedFlatFile,
    article: str,
    k: int,
) -> str:
    # Parse the given article and get its links
    article_entry = wiki_links.get(article)
    article_entry = json.loads(article_entry)
    article_links = article_entry["links"]

    # Find top k links, plus the (k+1)th link, skipping Wiki-specific links
    valid_links = (
        x["target"] for x in article_links if not is_special(x["target"])
    )
    k_links = itertools.islice(valid_links, k)
    k_plus_one = next(valid_links, None)

    # Produce query
    return f"In the context of {', '.join(k_links)}, tell me about {k_plus_one}."

# Identify if the given link is a special Wikipedia-specific one
def is_special(target: str) -> bool:
    return (
        re.match(
            r"^(File|Category|Wikipedia|Template|Help|Draft|Portal|Book|Module|TimedText|MediaWiki|Special|Talk):",
            target,
        )
        is not None
    )

def main(
    index_file: str,
    data_file: str,
    article: str | None,
    k: int | None,
):
    """Takes the top k links from an article and generates a query of the form
    "In the context of link_1, link_2, link_3, ..., and link_k, tell me about
    link_(k+1)".
    If the article and k are not provided, then they will be read from stdin,
    where every line should be of the following form:
        Article Name:k
    """

    # If we have provided an article, we must also provide a k
    if k is None and article is not None:
        print("Provide k")
        print()
        sys.exit(1)

    # Read the database of Wikipedia articles and get the links
    wiki_links = IndexedFlatFile(index_file, data_file, show_progress=True)

    # If a specific article has been provided, generate a query for that article
    if article is not None:
        query = generate_query(wiki_links, article, k)
        print(query)

    # Otherwise, read article names from standard input
    else:
        try:
            for line in sys.stdin:
                try:
                    article, k = line.strip().split(":")
                    k = int(k)
                    query = generate_query(wiki_links, article, k)
                    print(query)
                except Exception as e:
                    print(f"Error: {e}")
        except KeyboardInterrupt:
            return

if __name__ == "__main__":
    fire.Fire(main)
