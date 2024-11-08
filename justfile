set dotenv-load

_default:
	just -l

cv_hier_db_out_dir := "../rigel-data/context-vectors-compressed" 
cv_hier_db_test_out_dir := "../rigel-data/context-vectors-compressed-test" 

_cv_hier_db mode out_dir=cv_hier_db_out_dir narrow_factor='16' limit='':
	python3 -m generate_hier_cv_db \
		 	--compressor_chkpt "../rigel-data/hierarchical-compression-checkpoint/attention_model.pt" \
			--cv_db_dir "../rigel-data/context-vectors/" \
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

query_generator article k:
	python3 -m query_generator $WIKIPEDIA_LINKS_INDEX $WIKIPEDIA_LINKS_DATA "{{article}}" "{{k}}"

tests:
	python3 -m cv_storage.tests
	python3 -m wikipedia_parser.tests
	python3 -m cv_hier_storage.tests
