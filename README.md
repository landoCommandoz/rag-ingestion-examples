# RAG ingestion examples: PDFs and recordings, one chunk schema, real citations

Working ingestion pipelines for two pay-per-use Apify Actors that return the **same chunk schema**:

- [pdf-to-json-tables](https://apify.com/landoz/pdf-to-json-tables): PDF pages to text, Markdown, tables with cell coordinates, and RAG chunks with **page citations**.
- [transcribe-to-json-chunks](https://apify.com/landoz/transcribe-to-json-chunks): audio and video to transcript, timestamped segments, subtitles, and RAG chunks with **time citations**.

Because both Actors emit identical chunk objects, one loader indexes a 40-page report and a 40-minute recording the same way, and every answer can point at "p. 3" or "00:12:03-00:12:48" instead of just a file name.

Examples here: plain Python, LangChain, LlamaIndex and n8n. The default embeddings are local and keyless (all-MiniLM-L6-v2 or BM25). The only credential any example needs is your own Apify token, read from the `APIFY_TOKEN` environment variable.

This is an independent community repository by the author of the two Actors. It is not official Apify material and is not affiliated with or endorsed by Apify.

## The shared chunk schema

Every chunk, from either Actor:

```json
{
  "chunkIndex": 0,
  "text": "...",
  "charStart": 0,
  "charEnd": 1187,
  "citation": { "type": "page" | "time", "label": "..." }
}
```

A PDF chunk (also carries `page` at the top level):

```json
{
  "chunkIndex": 1,
  "page": 2,
  "charStart": 460,
  "charEnd": 1659,
  "text": "Oct 14, 2026 Oct 22, 2025\nReserve Bank credit 6,706,861 - 789 + 161,908 6,706,851\nSecurities held outright1 6,470,860 + ...",
  "citation": { "type": "page", "page": 2, "label": "p. 2" }
}
```

A transcript chunk (also carries `startMs` and `endMs` at the top level):

```json
{
  "chunkIndex": 1,
  "text": "r above our poor power to add or detract. The world will little note no longer remember what we say here, but it can never forget what they did here ...",
  "charStart": 928,
  "charEnd": 1682,
  "startMs": 50640,
  "endMs": 97840,
  "citation": { "type": "time", "startMs": 50640, "endMs": 97840, "label": "00:00:51-00:01:38" }
}
```

`charStart` and `charEnd` are offsets into the page text (PDF) or the full transcript (audio). Full dataset items, trimmed from real runs, are in [`samples/`](samples/).

## Quick start (5 minutes)

1. Get an Apify API token: [Apify docs, API token](https://docs.apify.com/platform/integrations/api#api-token). New accounts include free monthly usage; the test inputs below cost a few cents at list price.
2. Put it in your shell, never in a file in this repo:
   ```bash
   export APIFY_TOKEN=...   # macOS and Linux
   ```
3. Install and run the plain-Python examples:
   ```bash
   git clone <this repo> && cd rag-ingestion-examples
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r python/requirements.txt
   python python/basic_pipeline.py          # runs both Actors, writes out/chunks.jsonl
   python python/search_local.py "What affects the reserve balances of depository institutions?"
   ```
   Or `./run_all.sh` (or `make venv all`) does the same.
4. Optional, better search: `pip install -r python/requirements-embeddings.txt` adds local sentence embeddings. `search_local.py` picks them up automatically. On Linux install the CPU-only PyTorch wheel first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) to avoid a 2 GB download.

What you should see from `basic_pipeline.py` (sample PDF pages 1-2 plus the sample 100 second MP3):

```
PDF:   2 page(s), 4 chunk(s), charged {'page-processed': 2, 'ocr-page': 0}, list price $0.006
Audio: 2 started minute(s), 2 chunk(s), charged {'audio-minute-standard': 2, 'audio-minute-accurate': 0}, list price $0.024
Wrote 6 records to .../out/chunks.jsonl
```

## The examples

| Example | What it shows | Install | Status |
| --- | --- | --- | --- |
| [`python/basic_pipeline.py`](python/basic_pipeline.py) | Runs both Actors with `apify-client`, merges every chunk into `{id, text, source_url, citation_label, citation}` records, writes `out/chunks.jsonl` | `python/requirements.txt` | Ran end to end 2026-10-23 |
| [`python/search_local.py`](python/search_local.py) | Keyless search over the JSONL: all-MiniLM-L6-v2 embeddings when installed, pure-Python BM25 otherwise; prints the top 3 chunks with their `citation.label` | `python/requirements.txt` (+ `requirements-embeddings.txt` for MiniLM) | Ran end to end 2026-10-23, both engines |
| [`python/langchain_example.py`](python/langchain_example.py) | `ApifyFamilyChunkLoader(BaseLoader)` yields one `Document` per chunk with citation metadata; in-memory Chroma store with local `HuggingFaceEmbeddings`; similarity search prints citations | `python/requirements-langchain.txt` | Ran end to end 2026-10-23 |
| [`python/llamaindex_example.py`](python/llamaindex_example.py) | `ApifyFamilyChunkReader(BaseReader)` returns `Document`s with the same metadata; `VectorStoreIndex` with local `HuggingFaceEmbedding`; retriever only, so no LLM key | `python/requirements-llamaindex.txt` | Ran end to end 2026-10-23 |
| [`n8n/rag-ingest-family-chunks.json`](n8n/rag-ingest-family-chunks.json) | Importable n8n workflow: Manual Trigger, Set URLs, two HTTP Request nodes calling the Actors' `run-sync-get-dataset-items` endpoints, Merge, a Code node that flattens `chunks[]` with `citation.label`, and a final node to attach your vector store | n8n 1.x | Written and validated as JSON; not executed (no n8n in the test environment) |

All Python examples were run against the live Actors from a clean virtual environment with the pinned versions in the requirements files (Python 3.13, Linux, CPU only). Framework imports change often; if an import breaks, the pinned versions are the ones that worked.

### Metadata the loaders attach

Both the LangChain loader and the LlamaIndex reader put the same keys on every chunk, so a vector store filter written for PDFs works for recordings too:

| Key | PDF chunk | Transcript chunk |
| --- | --- | --- |
| `source` | PDF URL | media URL |
| `document_id` | sha1 of the URL | sha1 of the URL |
| `citation_type` | `page` | `time` |
| `citation_label` | `p. 2` | `00:00:51-00:01:38` |
| `page` | 2 | absent |
| `start_ms`, `end_ms` | absent | 50640, 97840 |
| `chunk_index`, `char_start`, `char_end` | as in the chunk | as in the chunk |

## Cite the source

The point of the shared schema is that `citation.label` is ready to show. An answer built from retrieved chunks can end with the labels of the chunks it used:

```python
hits = store.similarity_search(question, k=3)          # LangChain; same idea in LlamaIndex or plain Python
sources = sorted({f"{h.metadata['source'].rsplit('/', 1)[-1]}, {h.metadata['citation_label']}" for h in hits})
answer += "\n\nSources: " + "; ".join(sources)
# Sources: h41.pdf, p. 2; Gettysburg_Address_Lincoln_64kb.mp3, 00:00:51-00:01:38
```

For a PDF, `page` lets a viewer jump to the page. For a recording, `start_ms` lets a player seek to the moment (`startMs / 1000` seconds), and the label already reads as a time range. If you need a tighter quote than a chunk, both Actors return the finer units too: tables with cell coordinates on the PDF side, timestamped segments (and optionally words) on the audio side.

## What it costs

Both Actors are pay per event, billed by Apify to your account, with compute included:

| Actor | Price | Example |
| --- | --- | --- |
| pdf-to-json-tables | $0.003 per page processed, plus $0.01 per page only when OCR runs | a 30-page report: $0.09 |
| transcribe-to-json-chunks | $0.012 per started minute (standard tier), $0.035 per minute (accurate tier) | a 45-minute episode: $0.54 standard, $1.58 accurate |

Files that cannot be fetched or read return an error item and are not charged. The test inputs used in this repo (2 PDF pages and a 100 second recording) cost $0.03 per full run of `basic_pipeline.py` at list price. You can cap any run with `max_total_charge_usd` in `apify-client` or `maxTotalChargeUsd` on the REST API.

## n8n

n8n could not be run in the environment where these examples were tested, so the workflow is validated as well-formed n8n JSON (nodes, connections, settings) but not executed. To use it:

1. In n8n open **Workflows**, choose **Import from File**, and pick `n8n/rag-ingest-family-chunks.json`.
2. Make `APIFY_TOKEN` available: either set it as an environment variable where n8n runs (self-hosted; the HTTP Request nodes read `{{ $env.APIFY_TOKEN }}`, which needs `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`), or on n8n Cloud open each HTTP Request node, switch Authentication to a **Header Auth** credential with name `Authorization` and value `Bearer <your token>`, and delete the expression header.
3. Edit the **Set source URLs** node with your own PDF and media URLs, then press **Test workflow**.

The two HTTP Request nodes call `POST https://api.apify.com/v2/acts/landoz~pdf-to-json-tables/run-sync-get-dataset-items` and `POST https://api.apify.com/v2/acts/landoz~transcribe-to-json-chunks/run-sync-get-dataset-items` with a 300 second timeout. The sync endpoints stop waiting after 300 seconds, which covers a few dozen PDF pages or about 10 minutes of audio on the standard tier. For longer recordings start the run with `POST /v2/acts/<actor>/runs`, poll `GET /v2/actor-runs/<runId>` until `SUCCEEDED`, then read `GET /v2/datasets/<defaultDatasetId>/items`.

## Use the Actors from an AI agent

Both Actors are listed on the [Apify MCP server](https://mcp.apify.com). An agent connected to it can find them with `search-actors` ("pdf to json", "audio transcription json"), read their input schema with `fetch-actor-details`, run them with `call-actor`, and read `chunks[]` with `get-dataset-items`. The chunk schema above is what comes back.

## Repository layout

```
README.md
LICENSE                          MIT
Makefile, run_all.sh             run the plain-Python examples in order
python/
  basic_pipeline.py              both Actors to out/chunks.jsonl
  search_local.py                keyless search with citations
  langchain_example.py           BaseLoader + Chroma + local embeddings
  llamaindex_example.py          BaseReader + VectorStoreIndex + local embeddings
  requirements*.txt              pinned versions per example
n8n/rag-ingest-family-chunks.json
samples/                         trimmed real output from each Actor
```

## Notes

- Tokens stay in the environment. Nothing in this repo reads a key from a file, and `.env` is ignored by git.
- The sample inputs are public documents: a Federal Reserve statistical release and a public-domain LibriVox recording.
- Issues with the Actors themselves belong in the Issues tab of each Actor in Apify Console.

## License

MIT, see [LICENSE](LICENSE). Copyright 2026 Landon Brewington.
