set dotenv-load

_default:
	just -l

query_generator article k:
	python -m query_generator $WIKIPEDIA_LINKS_INDEX $WIKIPEDIA_LINKS_DATA "{{article}}" "{{k}}"

tests:
	python -m cv_storage.tests
	python -m wikipedia_parser.tests
