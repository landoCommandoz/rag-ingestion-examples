# Plain-Python examples in order. Needs APIFY_TOKEN in the environment.
#   make venv      create .venv and install the basic requirements
#   make all       run basic_pipeline.py then search_local.py
#   make langchain / make llamaindex   framework examples (install their requirements first)
#   make check     validate the n8n workflow JSON and scan for stray secrets

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
QUERY ?= What affects the reserve balances of depository institutions?

.PHONY: venv embeddings basic search all langchain llamaindex check clean

venv:
	python3 -m venv .venv
	$(PIP) install -r python/requirements.txt

embeddings:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r python/requirements-embeddings.txt

basic:
	$(PY) python/basic_pipeline.py

search:
	$(PY) python/search_local.py "$(QUERY)"

all: basic search

langchain:
	$(PY) python/langchain_example.py "$(QUERY)"

llamaindex:
	$(PY) python/llamaindex_example.py "$(QUERY)"

check:
	$(PY) -c "import json; w=json.load(open('n8n/rag-ingest-family-chunks.json')); assert {'nodes','connections','settings'} <= set(w); print('n8n workflow JSON ok:', len(w['nodes']), 'nodes')"
	@! grep -rIn --exclude-dir=.venv --exclude-dir=.git --exclude-dir=out -E "apify_api_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{20,}" . || (echo "possible secret found" && exit 1)
	@echo "no token-shaped strings in the repo"

clean:
	rm -rf out python/__pycache__
