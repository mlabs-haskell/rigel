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


def extract_section_text(section_titles: list[str], article: dict) -> str | None:
    # Sanity check
    section_title, *sub_sections = section_titles
    assert section_title == article["section_name"]

    # See if we're at the section we need
    if len(sub_sections) == 0:
        return article["text"]

    # Find applicable children and move down the tree through them
    for child in article["children"]:
        if child["section_name"] == sub_sections[0]:
            extracted_text = extract_section_text(sub_sections, child)
            if extracted_text is not None:
                return extracted_text

    # If we've gotten this far, we haven't found what we're looking for
    return None
