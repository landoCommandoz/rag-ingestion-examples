#!/usr/bin/env python3
"""Run both Actors and merge their chunks into one JSONL file.

Plain Python, apify-client only. Reads APIFY_TOKEN from the environment.

    python python/basic_pipeline.py                 # sample PDF pages 1-2 plus the sample MP3
    python python/basic_pipeline.py --pdf URL --audio URL --pages 1-5

Both Actors return chunks in the same schema, so one loop turns PDF pages and
recordings into identical records:

    {"id": ..., "text": ..., "source_url": ..., "citation_label": ..., "citation": {...}}

Output: out/chunks.jsonl (one record per line), ready for any vector store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from apify_client import ApifyClient

PDF_ACTOR = "landoz/pdf-to-json-tables"
AUDIO_ACTOR = "landoz/transcribe-to-json-chunks"

SAMPLE_PDF = "https://www.federalreserve.gov/releases/h41/current/h41.pdf"
SAMPLE_AUDIO = "https://archive.org/download/gettysburg_shurtagal_librivox/Gettysburg_Address_Lincoln_64kb.mp3"

OUT_PATH = Path(__file__).resolve().parent.parent / "out" / "chunks.jsonl"


def get_client() -> ApifyClient:
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        sys.exit(
            "APIFY_TOKEN is not set. Create a token in Apify Console (Settings, API and "
            "integrations) and run: export APIFY_TOKEN=...   (never paste it into code)"
        )
    return ApifyClient(token)


def run_actor(client: ApifyClient, actor_id: str, run_input: dict) -> tuple[list[dict], dict]:
    """Start an Actor run, wait for it, and return (dataset items, run summary).

    apify-client 3.x returns a Run model (attribute access). If you are on
    apify-client 2.x the same values are run["defaultDatasetId"],
    run["chargedEventCounts"] and run["usageTotalUsd"].
    """
    print(f"Running {actor_id} ...", flush=True)
    run = client.actor(actor_id).call(run_input=run_input, logger=None)
    if run is None or run.status != "SUCCEEDED":
        status = run.status if run else "no run"
        sys.exit(f"{actor_id} did not succeed (status {status}). Check the run in Apify Console.")
    items = list(client.dataset(run.default_dataset_id).iterate_items())
    # Charged event counts are filled in a moment after the run finishes, so re-read the run.
    run = client.run(run.id).get() or run
    summary = {
        "actor": actor_id,
        "run_id": run.id,
        "charged_events": dict(run.charged_event_counts or {}),
        "list_price_usd": list_price_usd(run),
    }
    return items, summary


def list_price_usd(run) -> float | None:
    """What a buyer pays for this run: charged events times the Actor's event prices."""
    try:
        events = run.pricing_info.pricing_per_event.actor_charge_events
        counts = run.charged_event_counts or {}
        return round(sum(counts.get(name, 0) * ev.event_price_usd for name, ev in events.items()), 4)
    except AttributeError:
        return None


def chunk_records(items: list[dict], kind: str) -> list[dict]:
    """Flatten the chunks[] of every dataset item into uniform records.

    kind is "pdf" or "audio"; the record shape is the same for both.
    """
    records = []
    for item in items:
        if item.get("error"):
            print(f"  skipped {item.get('sourceUrl')}: {item['error']}", file=sys.stderr)
            continue
        doc_id = item["documentId"]
        for chunk in item.get("chunks", []):
            citation = chunk["citation"]
            if citation["type"] == "page":
                rec_id = f"{doc_id}:p{citation['page']}:c{chunk['chunkIndex']}"
            else:
                rec_id = f"{doc_id}:t{citation['startMs']}:c{chunk['chunkIndex']}"
            records.append(
                {
                    "id": rec_id,
                    "text": chunk["text"],
                    "source_url": item["sourceUrl"],
                    "citation_label": citation["label"],
                    "citation": citation,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", default=SAMPLE_PDF, help="public PDF URL")
    parser.add_argument("--pages", default="1-2", help='pages to parse, like "1-2" or "3-"')
    parser.add_argument("--audio", default=SAMPLE_AUDIO, help="public audio or video URL")
    parser.add_argument("--tier", default="standard", choices=["standard", "accurate"], help="transcription tier")
    parser.add_argument("--out", default=str(OUT_PATH), help="where to write the JSONL")
    args = parser.parse_args()

    client = get_client()

    pdf_items, pdf_summary = run_actor(
        client, PDF_ACTOR, {"pdfUrls": [args.pdf], "pages": args.pages}
    )
    audio_items, audio_summary = run_actor(
        client, AUDIO_ACTOR, {"audioUrls": [args.audio], "model": args.tier}
    )

    records = chunk_records(pdf_items, "pdf") + chunk_records(audio_items, "audio")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pages = sum(1 for it in pdf_items if not it.get("error"))
    pdf_chunks = sum(len(it.get("chunks", [])) for it in pdf_items)
    audio_chunks = sum(len(it.get("chunks", [])) for it in audio_items)
    minutes = sum(it.get("durationMinutesBilled", 0) for it in audio_items)

    print()
    print(f"PDF:   {pages} page(s), {pdf_chunks} chunk(s), charged {pdf_summary['charged_events']}, list price ${pdf_summary['list_price_usd']}")
    print(f"Audio: {minutes} started minute(s), {audio_chunks} chunk(s), charged {audio_summary['charged_events']}, list price ${audio_summary['list_price_usd']}")
    print(f"Wrote {len(records)} records to {out}")
    if records:
        first, last = records[0], records[-1]
        print(f"First record: {first['citation_label']!r} {first['text'][:70]!r}...")
        print(f"Last record:  {last['citation_label']!r} {last['text'][:70]!r}...")


if __name__ == "__main__":
    main()
