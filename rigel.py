from compression.compressor import Compressor
from compression.loss_functions import sequence_similarity
from cv_storage.cv_hier_storage import ContextVectorHierDB, DBConfig

from modified_llama.llama import Llama

from pathlib import Path
import torch

class Rigel():
    def __init__(
        self,
        llama_checkpoint_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        cv_db_dir: str,
        compression_checkpoint_file: str
    ):
        # Create the Llama generator
        print("Building generator...")
        self.generator = Llama.build(
            ckpt_dir=llama_checkpoint_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=1,
        )
        print("Built generator!")

        torch.set_default_dtype(torch.float32)
        config = DBConfig([16, 64, 256, 1024, 4096])
        self.cv_db = ContextVectorHierDB(Path(cv_db_dir), config, sequence_similarity)
        self.compressor = Compressor(compression_checkpoint_file)

        self.max_seq_len = max_seq_len

    def generate(self, query: str) -> str:
        # Tokenize the query, and generate context vectors for it
        query_tokens = self.generator.tokenize(self.max_seq_len, [("", query)])
        _, query_tokens = query_tokens[0]
        query_cv = self.generator.generate_context_vectors([query_tokens[1:]], 0, 0)

        with torch.no_grad():
            # Compress the context vector
            query_cv = query_cv.to(torch.float32)
            compressed_cvs = self.compressor.compress(query_cv)
            compressed_cvs = [query_cv, *compressed_cvs]
            compressed_cvs.reverse()

        # Search the database for the most relevant content
        results = self.cv_db.search(compressed_cvs, 16)
        content_cv = results[0].cv.to(torch.float16).unsqueeze(0)

        # Yield the generated text
        tokens, _ = self.generator.generate([query_tokens], content_cv, 0)
        tokens = tokens[0]
        output = self.generator.tokenizer.decode(tokens)
        return " ".join([query, output])