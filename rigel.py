from compression.compressor import Compressor
from compression.loss_functions import sequence_similarity
from cv_storage.cv_hier_storage import ContextVectorHierDB, DBConfig
from wikipedia_parser import IndexedFlatFile
from wikipedia_parser.articles import generate_texts

from modified_llama.llama import Llama

import json
from pathlib import Path
import torch

class Rigel():
    def __init__(
        self,
        llama_checkpoint_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        cv_db_dir: str,
        compression_checkpoint_file: str,
        content_index_file: str,
        content_data_file: str,
        extra_docs_dir: str = "extra_docs"
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

        # Create context vector database
        torch.set_default_dtype(torch.float32)
        config = DBConfig([16, 64, 256, 1024, 4096])
        self.cv_db = ContextVectorHierDB(Path(cv_db_dir), config, sequence_similarity)
        self.compressor = Compressor(compression_checkpoint_file)
        self.articles_db = IndexedFlatFile(content_index_file, content_data_file)

        # Get extra documents
        self.extra_docs = {}
        extra_docs_dir = Path(extra_docs_dir)
        if extra_docs_dir.exists():
            for path in extra_docs_dir.iterdir():
                if path.is_file():
                    with open(path, "r") as file:
                        article = json.load(file)

                    article_title = article["section_name"]
                    self.extra_docs[article_title] = {}
                    for section_name, text in generate_texts(article):
                        self.extra_docs[article_title][section_name] = text

        self.max_seq_len = max_seq_len

    def generate_llama(self, query:str, temperature: float = 0.6) -> str:
        # Tokenize the query
        query_tokens = self.generator.tokenize(self.max_seq_len, [("", query)])
        _, query_tokens = query_tokens[0]

        # Yield the generated text
        tokens, _ = self.generator.generate([query_tokens], temperature=temperature)
        tokens = tokens[0]
        output = self.generator.tokenizer.decode(tokens)
        return " ".join([query, output])

    def generate(self, query: str, verbose: bool = False, temperature: float = 0.6) -> str:
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
        results = self.cv_db.search(compressed_cvs, 4)
        metadata = self.cv_db.get_metadata(results[0].idx)
        if verbose:
            print(f"\t-- (Relevant content: {metadata.article_title})")

        # Get original content
        title = metadata.article_title
        section = metadata.section_name
        if title in self.extra_docs and section in self.extra_docs[title]:
            text = self.extra_docs[title][section]
        else:
            json_str = self.articles_db.get(title)
            obj = json.loads(json_str)
            text = find_section(section, obj)

        # Generate RAG query
        new_query = text + "\n" + query
        query_tokens = self.generator.tokenize(self.max_seq_len, [("", new_query)])
        _, query_tokens = query_tokens[0]

        # Yield the generated text
        tokens, _ = self.generator.generate([query_tokens], temperature=temperature)
        tokens = tokens[0]
        output = self.generator.tokenizer.decode(tokens)
        return " ".join([query, output])

def find_section(section_name: str, obj: dict[str]) -> str:
    def worker(sections: list[str], obj: dict[str]) -> str:
        if sections[0] == "root":
            sections = sections[1:]

        if len(sections) == 0:
            return obj["text"]

        next_child = sections[0]
        for child in obj["children"]:
            if child["section_name"] == next_child:
                return worker(sections[1:], child)

        raise Exception("This shouldn't happen")

    sections = section_name.split("\\")
    return worker(sections, obj)