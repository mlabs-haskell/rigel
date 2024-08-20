from cv_library.hierarchical_compression import HierarchicalAttention
from cv_library.loss_functions import sequence_similarity
from cv_library.train import DEVICE

import fire
from modified_llama.llama import Llama
from pathlib import Path
import torch

def evaluate(
    llama_checkpoint_dir: str,
    tokenizer_path: str,
    compression_checkpoint_dir: str,
    content_string: str,
    query_string: str,
    max_seq_len: int = 128,
    max_batch_size: int = 4
):
    # Create the Llama generator
    print("Building generator...")
    generator = Llama.build(
        ckpt_dir=llama_checkpoint_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("Built generator!")

    # Tokenize the content and query, and generate context vectors for them
    content_tokens = generator.tokenize(max_seq_len, [("", content_string)])
    query_tokens = generator.tokenize(max_seq_len, [("", query_string)])
    content_cvs = generator.generate([l for _, l in content_tokens])
    query_cvs = generator.generate([l for _, l in query_tokens])

    # Make sure the generator isn't in GPU memory anymore
    del generator

    # Load the compression network
    with torch.device(DEVICE):
        compression_network = HierarchicalAttention(4096)
        checkpoint_path = Path(compression_checkpoint_dir)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path)
            compression_network.load_state_dict(checkpoint['model_state_dict'])

        compressed_content = compression_network.forward(content_cvs)
        compressed_query = compression_network.forward(query_cvs)
        sim_score = sequence_similarity(compressed_content, compressed_query)

    print(sim_score)

if __name__ == "__main__":
    fire.Fire(evaluate)