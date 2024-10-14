import itertools
import json
from pathlib import Path
import re

from wikipedia_parser import IndexedFlatFile


def generate_query(
    wiki_links: IndexedFlatFile,
    article: str,
    k: int,
) -> str:
    article_entry = wiki_links.get(article)
    article_entry = json.loads(article_entry)

    article_links = article_entry["links"]

    valid_links = (
        x["target"] for x in article_links if not is_special(x["target"])
    )  # iterator of links which are not special
    k_links = itertools.islice(valid_links, k)  # take k links
    k_plus_one = next(valid_links, None)  # take the k+1th valid link

    return f"In the context of {', '.join(k_links)}, tell me about {k_plus_one}."


def is_special(target: str) -> bool:
    return (
        re.match(
            r"^(File|Category|Wikipedia|Template|Help|Draft|Portal|Book|Module|TimedText|MediaWiki|Special|Talk):",
            target,
        )
        is not None
    )
