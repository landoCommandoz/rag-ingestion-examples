#!/usr/bin/env python3
"""LangChain: a document loader for both Actors, a local Chroma vector store, cited search.

    pip install -r python/requirements-langchain.txt
    export APIFY_TOKEN=...
    python python/langchain_example.py "What affects reserve balances?"

ApifyFamilyChunkLoader runs one Actor (pdf-to-json-tables or transcribe-to-json-chunks)
and yields one LangChain Document per chunk. Because both Actors share the chunk schema,
the same loader class handles PDFs and recordings; only the Actor id and input differ.

Embeddings are local (all-MiniLM-L6-v2 through sentence-transformers), the vector store is
an in-memory Chroma collection, and no LLM is called, so APIFY_TOKEN is the only key needed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

from apify_client import ApifyClient
from langchain_chroma import Chroma
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

PDF_ACTOR = "landoz/pdf-to-json-tables"
AUDIO_ACTOR = "landoz/transcribe-to-json-chunks"
SAMPLE_PDF = "https://www.federalreserve.gov/releases/h41/current/h41.pdf"
SAMPLE_AUDIO = "https://archive.org/download/gettysburg_shurtagal_librivox/Gettysburg_Address_Lincoln_64kb.mp3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ApifyFamilyChunkLoader(BaseLoader):
    """Load the chunks[] of an Apify Actor run as LangChain Documents.

    Works with any Actor that emits the shared chunk schema
    {chunkIndex, text, charStart, charEnd, citation: {type, label, page | startMs, endMs}}.
    """

    def __init__(self, actor_id: str, run_input: dict, client: ApifyClient | None = None):
        self.actor_id = actor_id
        self.run_input = run_input
        self.client = client or ApifyClient(os.environ["APIFY_TOKEN"])

    def lazy_load(self) -> Iterator[Document]:
        run = self.client.actor(self.actor_id).call(run_input=self.run_input, logger=None)
        if run is None or run.status != "SUCCEEDED":
            raise RuntimeError(f"{self.actor_id} did not succeed: {run.status if run else 'no run'}")
        for item in self.client.dataset(run.default_dataset_id).iterate_items():
            if item.get("error"):
                print(f"skipped {item.get('sourceUrl')}: {item['error']}", file=sys.stderr)
                continue
            for chunk in item.get("chunks", []):
                yield Document(page_content=chunk["text"], metadata=self._metadata(item, chunk))

    @staticmethod
    def _metadata(item: dict, chunk: dict) -> dict:
        citation = chunk["citation"]
        meta = {
            "source": item["sourceUrl"],
            "document_id": item["documentId"],
            "citation_type": citation["type"],
            "citation_label": citation["label"],
            "chunk_index": chunk["chunkIndex"],
            "char_start": chunk["charStart"],
            "char_end": chunk["charEnd"],
        }
        if citation["type"] == "page":
            meta["page"] = citation["page"]
        else:
            meta["start_ms"] = citation["startMs"]
            meta["end_ms"] = citation["endMs"]
        return meta


def main() -> None:
    if not os.environ.get("APIFY_TOKEN"):
        sys.exit("APIFY_TOKEN is not set.")
    query = sys.argv[1] if len(sys.argv) > 1 else "What affects the reserve balances of depository institutions?"

    print("Loading chunks from both Actors ...")
    docs = ApifyFamilyChunkLoader(PDF_ACTOR, {"pdfUrls": [SAMPLE_PDF], "pages": "1-2"}).load()
    docs += ApifyFamilyChunkLoader(AUDIO_ACTOR, {"audioUrls": [SAMPLE_AUDIO], "model": "standard"}).load()
    print(f"{len(docs)} documents loaded")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma.from_documents(docs, embeddings, collection_name="family_chunks")  # in memory

    print(f"\nQuery: {query}\n")
    for doc, distance in store.similarity_search_with_score(query, k=3):
        m = doc.metadata
        where = f"page {m['page']}" if m["citation_type"] == "page" else f"{m['start_ms']} to {m['end_ms']} ms"
        print(f"[{m['citation_label']}]  distance {distance:.3f}  ({where})  {m['source']}")
        print("   ", " ".join(doc.page_content.split())[:150], "...\n")

    best = store.similarity_search(query, k=1)[0].metadata
    print(f'Cite it as: "{best["source"].rsplit("/", 1)[-1]}, {best["citation_label"]}"')


if __name__ == "__main__":
    main()
