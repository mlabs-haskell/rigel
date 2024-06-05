DB_DIR = "db"

# The folder which contains bin_xx.bson.gz files
DUMPS_DIR = "../../../dump/dl/"

# The folder which contains *.txt file where each file contains the list of
# article titles belonging to a subgraph.
# The filename of the text file is the name of the root of the subgraph.
GROUPS_DIR = "groups"

# store outputs here
OUT_DIR = "out"

# These items have context vectors with length != 524288
# Every other entry has the same length
EXCLUDE_IDS = [
    "663dc164bcbca1a4f0b7454b",
    "663dad03bcbca1a4f0b706ad",
    "663dad03bcbca1a4f0b706ae",
    "663eacc73a78a51401fefc0e",
    "663eacc73a78a51401fefc0f",
    "663eacc73a78a51401fefc10",
    "663eacc73a78a51401fefc11",
    "663eacc73a78a51401fefc12",
    "663eacc73a78a51401fefc13",
    "663eacc73a78a51401fefc14",
    "663eacc73a78a51401fefc15",
    "663eacc83a78a51401fefc16",
    "663ed24a3a78a51401ff2b28",
    "663d9e405b70b24ea34c0ef6",
    "663da285bcbca1a4f0b6db24",
]

# Calculate similarity based on a random sample of this size
SAMPLE_SIZE = 25
