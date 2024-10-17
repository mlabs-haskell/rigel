from typing import Iterator


def generate_texts(article) -> Iterator[tuple[str, str]]:
    # Generate text per section
    def get_text_by_section(section_name, article):
        if section_name == "":
            section_name = "root"
        else:
            subsection_name = article["section_name"]
            section_name = f"{section_name}\\{subsection_name}"

        yield (section_name, article["text"])

        for child in article["children"]:
            yield from get_text_by_section(section_name, child)

    yield from get_text_by_section("", article)


def extract_section_text(section_title: str, article: dict) -> str:
    if section_title == "root" or section_title == article["section_name"]:
        return article["text"]

    key, *rest = section_title.split("\\", maxsplit=1)
    if len(rest) == 0:
        return None

    for child in article["children"]:
        if child["section_name"].startswith(key):
            return extract_section_text(rest[0], child)
