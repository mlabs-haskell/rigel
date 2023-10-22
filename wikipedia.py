import mwxml
import re

# Constants
FILE = "enwiki-20231020-pages-articles-multistream1.xml-p1p41242"
REGEX = re.compile("{{[^{}]*}}")

# Store article sections
class Tree():
    def __init__(self, text: str, children: dict[str, any]):
        self.children = children
        self.text = text

    def print(self, level: int = 0):
        dashes = "-" * level
        print(f"{dashes} {self.text}")
        for key, val in self.children.items():
            print(f"{dashes}| {key}")
            val.print(level + 1)

# Get the heading text from a subject header
def parse_header(text: str) -> tuple[list[str], int]:
    level = 0
    for c in text:
        if c == "=":
            level += 1
        else:
            break
    return (text[level:-level].strip(), level)

# Recursively parse the article
def parse_article(lines: list[str], level: int | None = None) -> tuple[Tree, list[str]]:
    text = []
    children = {}
    while len(lines) > 0:
        line = lines[0]

        # Handle header lines
        if line.startswith("=="):
            header, new_level = parse_header(line)

            # Parse child headings
            if level is None or new_level > level:
                child, lines = parse_article(lines[1:], new_level)
                children[header] = child

            # Stop parsing. Next block is sibling or ancestor
            else:
                break
        
        # Accumulate text lines
        else:
            text.append(line)
            lines = lines[1:]   

    return (Tree("\n".join(text), children), lines)

# Return the article, as a filtered series of lines
def clean_article(text: str) -> list[str]:
    new_text = re.sub(REGEX, "", text)
    new_text = new_text.split("\n")
    new_text = [t for t in new_text if len(t) > 0]
    return new_text

# Get the title and proccessed text of each article
pages = {}
dump = mwxml.Dump.from_file(FILE)
for page in dump.pages:
    if page.redirect is None:
        title = page.title
        for revision in page:
            text = revision.text
            break

        text = clean_article(text)
        tree, _ = parse_article(text)
        print(title, ":", list(tree.children.keys()))

        pages[title] = tree

#print(pages)