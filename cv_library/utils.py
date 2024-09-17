import json

# Reads JSON object from data file starting at given index
def read_data(contents_data_file: str, index: int) -> dict:
    with open(contents_data_file, 'r') as file:
        # Go to index in file
        file.seek(index)

        # Read JSON string one char at a time, keeping track of open braces
        buffer = ""
        num_open_braces = 0
        while True:
            c = file.read(1)
            if c == '{':
                num_open_braces += 1
            if c == '}':
                num_open_braces -= 1
            buffer += c

            if num_open_braces == 0:
                break

    ret_val = json.loads(buffer)
    return ret_val

def get_contents_indices(contents_index_file: str) -> dict[str, int]:
    contents_indices = {}
    with open(contents_index_file, 'r') as file:
        for line in file:
            line = line.strip()
            [index, article_name] = line.split(': ', maxsplit=1)
            contents_indices[article_name] = int(index)

    return contents_indices