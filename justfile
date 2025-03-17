set dotenv-load

_default:
	just -l

cv_hier_db_out_dir := "./data/context-vectors-compressed"
cv_hier_db_test_out_dir := "./data/context-vectors-compressed-test"

_cv_hier_db mode out_dir=cv_hier_db_out_dir narrow_factor='16' limit='':
	python3 -m scripts.generate_hier_cv_db \
			--compressor_chkpt "./data/attention_model.pt" \
			--cv_db_dir "./data/context-vectors/" \
			--hier_db_dir {{out_dir}} \
			--level_sizes 16,64,256,1024,4096 \
			--mode {{mode}} \
			{{ if limit != "" { "--limit " + limit } else { "" } }} \
			--search_narrow_factor {{narrow_factor}}

# Generate the hierarchical cv db
cv_hier_db_gen limit='':
	just _cv_hier_db "generate" {{ limit }}

# Do some sanity checks on the hierarchical cv db
cv_hier_db_verify limit='':
	just _cv_hier_db "verify" {{ limit }}

# Clean the heirarchical cv db
cv_hier_db_clean out_dir=cv_hier_db_out_dir:
	rm -rf {{out_dir}}

cv_hier_db_test_e2e:
	just cv_hier_db_clean {{cv_hier_db_test_out_dir}}
	just _cv_hier_db "generate" {{cv_hier_db_test_out_dir}} 2 1000
	just _cv_hier_db "verify" {{cv_hier_db_test_out_dir}} 2 1000

generate_context_vectors:
	python3 -m scripts.generate_context_vectors \
			--ckpt_dir modified_llama/llama-2-7b-chat/ \
			--tokenizer_path modified_llama/tokenizer.model \
			--content_data_file ../wikipedia_parser/output/contents/data.jsonl \
			--content_index_file ../wikipedia_parser/output/contents/index.txt \
			--cv_db_folder ./data/context-vectors \
			--article_list_file all.txt \
			--extra_docs_dir extra_docs

generate_tfidf:
	python3 -m scripts.generate_tfidf \
			--contents_index_file ../wikipedia_parser/output/contents/index.txt \
			--contents_data_file ../wikipedia_parser/output/contents/data.jsonl \
			--out_file ./data/tfidf.json \
			--cvdb_folder ./data/context-vectors \
			--extra_docs_dir ./extra_docs

query_generator article k:
	python3 -m query_generator $WIKIPEDIA_LINKS_INDEX $WIKIPEDIA_LINKS_DATA "{{article}}" "{{k}}"

tests:
	python3 -m pytest

train_compressor validation='True' epochs='100':
	python3 -m scripts.train_compressor train \
			--checkpoint_file ./data/attention_model.pt \
			--network_type attention \
			--reduction_factor 4 \
			--cvdb_folder ./data/context-vectors/ \
			--validation {{validation}} \
			--epochs {{epochs}} \
			--tfidf_file ./data/tfidf.json

train_compressor_min_loss model='./data/attention_model.pt':
	python3 -m scripts.train_compressor min_loss \
			--checkpoint_file {{model}}
