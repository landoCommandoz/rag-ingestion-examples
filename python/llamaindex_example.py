#!/usr/bin/env python3
"""LlamaIndex: a reader for both Actors, a local VectorStoreIndex, cited retrieval.

    pip install -r python/requirements-llamaindex.txt
    export APIFY_TOKEN=...
    python python/llamaindex_example.py "What affects reserve balances?"

ApifyFamilyChunkReader runs one Actor and returns one LlamaIndex Document per chunk with
the citation in metadata. Embeddings are local (all-MiniLM-L6-v2 through
HuggingFaceEmbedding) and the index is queried through a retriever, so no LLM and no
LLM key are involved. APIFY_TOKEN is the only credential.
"""

from __future__ import annotations

import os
import sys

from apify_client import ApifyClient
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.readers.base import BaseReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PDF_ACTOR = "landoz/pdf-to-json-tables"
AUDIO_ACTOR = "landoz/transcribe-to-json-chunks"
SAMPLE_PDF = "https://www.federalreserve.gov/releases/h41/current/h41.pdf"
SAMPLE_AUDIO = "https://archive.org/download/gettysburg_shurtagal_librivox/Gettysburg_Address_Lincoln_64kb.mp3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ApifyFamilyChunkReader(BaseReader):
    """Return the chunks[] of an Apify Actor run as LlamaIndex Documents.

    Works with any Actor that emits the shared chunk schema
    {chunkIndex, text, charStart, charEnd, citation: {type, label, page | startMs, endMs}}.
    """

    def __init__(self, client: ApifyClient | None = None):
        self.client = client or ApifyClient(os.environ["APIFY_TOKEN"])

    def load_data(self, actor_id: str, run_input: dict) -> list[Document]:
        run = self.client.actor(actor_id).call(run_input=run_input, logger=None)
        if run is None or run.status != "SUCCEEDED":
            raise RuntimeError(f"{actor_id} did not succeed: {run.status if run else 'no run'}")
        docs: list[Document] = []
        for item in self.client.dataset(run.default_dataset_id).iterate_items():
            if item.get("error"):
                print(f"skipped {item.get('sourceUrl')}: {item['error']}", file=sys.stderr)
                continue
            for chunk in item.get("chunks", []):
                meta = self._metadata(item, chunk)
                docs.append(
                    Document(
                        text=chunk["text"],
                        metadata=meta,
                        doc_id=f"{item['documentId']}:{chunk['chunkIndex']}:{meta['citation_label']}",
                        # Embed the chunk text only; metadata is for display, not retrieval.
                        excluded_embed_metadata_keys=list(meta),
                        excluded_llm_metadata_keys=list(meta),
                    )
                )
        return docs

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

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    print("Loading chunks from both Actors ...")
    reader = ApifyFamilyChunkReader()
    docs = reader.load_data(PDF_ACTOR, {"pdfUrls": [SAMPLE_PDF], "pages": "1-2"})
    docs += reader.load_data(AUDIO_ACTOR, {"audioUrls": [SAMPLE_AUDIO], "model": "standard"})
    print(f"{len(docs)} documents loaded")

    # The Actors already chunked the text, so pass transformations=[] to keep one node per chunk.
    index = VectorStoreIndex.from_documents(docs, transformations=[], show_progress=False)
    retriever = index.as_retriever(similarity_top_k=3)

    print(f"\nQuery: {query}\n")
    nodes = retriever.retrieve(query)
    for node in nodes:
        m = node.metadata
        where = f"page {m['page']}" if m["citation_type"] == "page" else f"{m['start_ms']} to {m['end_ms']} ms"
        print(f"[{m['citation_label']}]  score {node.score:.3f}  ({where})  {m['source']}")
        print("   ", " ".join(node.text.split())[:150], "...\n")

    best = nodes[0].metadata
    print(f'Cite it as: "{best["source"].rsplit("/", 1)[-1]}, {best["citation_label"]}"')


if __name__ == "__main__":
    main()
