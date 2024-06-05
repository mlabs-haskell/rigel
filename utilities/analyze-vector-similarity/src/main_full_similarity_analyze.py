import json
from tqdm import tqdm
import sys

import config

output_filename = config.OUT_DIR + "/full_similarity.json"

lines = open(output_filename).readlines()
print("Loading data ..", file=sys.stderr)
data = [json.loads(line) for line in tqdm(lines)]

print("Sorting ..", file=sys.stderr)
data.sort(key=lambda x: x["similarity_index"], reverse=True)

# Remove some noise: similarity >=0.99 means they are the same article or the article is too short.
data = filter(lambda x: x["similarity_index"] < 0.99, data)

for item in data:
    print(item["x"]["title"] + "\\" + item["x"]["header"])
    print(item["y"]["title"] + "\\" + item["y"]["header"])
    print(item["similarity_index"])
    print()
