#!/usr/bin/env python3
"""Search the chunks written by basic_pipeline.py and print the top hits with citations.

Keyless by default. Two local engines are built in:

  minilm  sentence embeddings with all-MiniLM-L6-v2 (Apache 2.0), via sentence-transformers.
          Best quality. Needs: pip install -r python/requirements-embeddings.txt
          (about 90 MB of model weights are downloaded on first use).
  bm25    pure-Python BM25 keyword ranking via rank-bm25 (Apache 2.0). No model download,
          installs in a second, good enough to show the citation flow.

"auto" (the default) uses minilm when sentence-transformers is installed and falls back
to bm25 otherwise, and says which one it used.

An opt-in third engine, "openai", calls a hosted embeddings API and needs OPENAI_API_KEY
in the environment. It is never used unless you ask for it.

    python python/search_local.py "How much did reserve balances change?"
    python python/search_local.py --engine bm25 "four score and seven years ago"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "out" / "chunks.jsonl"
DEFAULT_QUERY = "What affects the reserve balances of depository institutions?"


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found. Run python/basic_pipeline.py first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- engines

def rank_minilm(query: str, texts: list[str]) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_vecs = model.encode(texts, normalize_embeddings=True)
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    return [float(v @ query_vec) for v in doc_vecs]  # cosine similarity, vectors are unit length


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rank_bm25(query: str, texts: list[str]) -> list[float]:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_tokens(t) for t in texts])
    return [float(s) for s in bm25.get_scores(_tokens(query))]


def rank_openai(query: str, texts: list[str]) -> list[float]:
    """Opt-in hosted embeddings. Requires OPENAI_API_KEY; uses only the standard library."""
    import urllib.request

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("--engine openai needs OPENAI_API_KEY in the environment.")
    body = json.dumps({"model": "text-embedding-3-small", "input": [query] + texts}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        vectors = [d["embedding"] for d in json.load(resp)["data"]]
    q, docs = vectors[0], vectors[1:]
    norm = lambda v: sum(x * x for x in v) ** 0.5
    return [sum(a * b for a, b in zip(q, d)) / (norm(q) * norm(d)) for d in docs]


ENGINES = {"minilm": rank_minilm, "bm25": rank_bm25, "openai": rank_openai}


def pick_engine(name: str) -> str:
    if name != "auto":
        return name
    try:
        import sentence_transformers  # noqa: F401
        return "minilm"
    except ImportError:
        return "bm25"


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--engine", default="auto", choices=["auto", *ENGINES])
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--chunks", default=str(CHUNKS_PATH))
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks))
    engine = pick_engine(args.engine)
    print(f"{len(chunks)} chunks loaded, engine: {engine}")
    print(f"Query: {args.query}\n")

    scores = ENGINES[engine](args.query, [c["text"] for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)[: args.top]

    for rank, (score, chunk) in enumerate(ranked, start=1):
        snippet = " ".join(chunk["text"].split())[:160]
        print(f"#{rank}  score {score:.3f}  [{chunk['citation_label']}]  {chunk['source_url']}")
        print(f"     {snippet}...\n")

    # This is the whole point: a citation an answer can show, built from citation.label.
    best = ranked[0][1]
    print(f'Answer footer example: "Source: {best["source_url"].rsplit("/", 1)[-1]}, {best["citation_label"]}"')


if __name__ == "__main__":
    main()
