_list:
	just -l

start-db:
	sudo systemctl start mongod

generate_context_vectors:
	RANK=0 \
	WORLD_SIZE=1 \
	MASTER_ADDR=127.0.0.1 \
	MASTER_PORT=2020 \
	python generate_context_vectors.py \
	modified_llama/llama-2-7b \
	modified_llama/tokenizer.model \
	../wikipedia_parser/output/contents/data.jsonl \
	../wikipedia_parser/output/contents/index.txt \
	../wikipedia_parser/output/subgraph/total.txt \
	--max_batch_size=16

