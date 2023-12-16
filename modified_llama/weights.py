import json
import math
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

ckpt_dir = "llama-2-7b"
tokenizer_path = "tokenizer.model"

checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"

print("Starting initial load")
ckpt_path = checkpoints[0]
checkpoint = torch.load(ckpt_path, map_location="cpu")
with open(Path(ckpt_dir) / "params.json", "r") as f:
    params = json.loads(f.read())
print("Finished initial load")

print("Moving to model")
model_args: ModelArgs = ModelArgs(
    max_seq_len=128,
    max_batch_size=4,
    **params,
)
tokenizer = Tokenizer(model_path=tokenizer_path)
model_args.vocab_size = tokenizer.n_words
model = Transformer(model_args)
model.load_state_dict(checkpoint, strict=False)
print("Moved to model")
del checkpoint

print("Counting params")
vals = {}
for p in model.parameters():
    nd = p.data.numpy()
    for val in nd.flat:
        val = abs(val)
        val = math.log10(val)
        val = round(val)

        if val not in vals:
            vals[val] = 0
            print(f"New val: {val}")
        vals[val] += 1

del model

print("Charting")
bins = list(vals.keys())
bins.sort()
counts = [math.log(vals[b]) for b in bins]

fig = plt.figure()
plt.bar(bins, counts)
plt.xlabel("Magnitude")
plt.ylabel("log(Count)")
plt.title("Distribution of Llama parameters")
plt.show()
