import sys

file = sys.argv[1]
start, end = int(sys.argv[2]), int(sys.argv[3])

with open(file, "rb") as f:
    f.seek(start)
    print(f.read(end - start).decode("utf-8"))
