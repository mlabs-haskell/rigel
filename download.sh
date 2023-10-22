# Download Wikipedia
WIKIPEDIA_LINK="https://dumps.wikimedia.org/enwiki/20231020/enwiki-20231020-pages-articles-multistream.xml.bz2"
curl $WIKIPEDIA_LINK | bzip2 -d > data/wikipedia.xml