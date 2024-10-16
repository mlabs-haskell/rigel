from dataclasses import dataclass
import sys
from query_generator import generate_query
from wikipedia_parser import IndexedFlatFile


@dataclass
class Args:
    index_file: str
    data_file: str
    article: str | None
    k: int | None


def parse_args() -> Args:
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    index_file: str = sys.argv[1]
    data_file: str = sys.argv[2]
    article: str | None = None
    k: int | None = None

    if len(sys.argv) > 3:
        article = sys.argv[3]
    if len(sys.argv) > 4:
        k_ = sys.argv[4]
        k = int(k_)

    return Args(
        index_file=index_file,
        data_file=data_file,
        article=article,
        k=k,
    )


def main():
    args = parse_args()

    if args.k is None and args.article is not None:
        print("Provide k")
        print()
        print_usage()
        sys.exit(1)

    wiki_links = IndexedFlatFile(args.index_file, args.data_file, show_progress=True)

    if args.article is not None:
        query = generate_query(wiki_links, args.article, args.k)
        print(query)
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


def print_usage():
    print("Usage:")
    print("  python -m query_generator <index_file> <data_file> <article> <k>")
    print("or")
    print(
        r"  echo '<article>:<k>\n<article>:<k>' | python -m query_generator <index_file> <data_file>"
    )


if __name__ == "__main__":
    main()
