# Default to config.toml if no config file is specified
.PHONY: all query_document_enrichment clean_and_embed index_and_retrieve table_selection column_rank

CONFIG_FILE ?= config_synt_template.toml

all: query_document_enrichment clean_and_embed index_and_retrieve table_selection column_rank

query_document_enrichment:
	python schema_matching/query_document_enrichment.py --config $(CONFIG_FILE)

clean_and_embed:
	python schema_matching/clean_and_embed.py --config $(CONFIG_FILE)

index_and_retrieve:
	python schema_matching/index_and_retrieve.py --config $(CONFIG_FILE)

table_selection:
	python schema_matching/table_selection.py --config $(CONFIG_FILE)

column_rank:
	python schema_matching/column_rank.py --config $(CONFIG_FILE)

# preprocess:
# 	@echo "Preprocessing is done"

# clean:
# 	rm -rf processed_data/*
