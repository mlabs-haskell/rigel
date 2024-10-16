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

