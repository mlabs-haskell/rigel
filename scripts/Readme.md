This folder contains scripts for generating context vectors and the associated
storage. Here is a description of each script in this folder:

- [generate_context_vectors.py](./generate_context_vectors.py)
  Use modified_llama to extract the context vectors from articles and store it using the [cv_storage](./cv_storage) library (see below).
  Check the arguments to the main function for the available options like input files and output folders.

- [generate_hier_cv_db.py](./generate_hier_cv_db.py)
  Read an instance of `cv_storage` and use `cv_library` to generate an instance of `cv_hier_storage`

- [query_generator.py](./query_generator.py)
  Generate a query that asks, in the context of the top k links from an article,
  to describe the (k+1)th link

- [text_processing.py](./text_processing.py)
  Generate a json file containing tfidf scores for each article. Used to train the
  hierarchical compression module

- [train_compressor](./train_compressor.py)
  Train a neural network that can hierarchically compress an input context vector