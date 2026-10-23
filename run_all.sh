#!/usr/bin/env bash
# Runs the plain-Python examples in order: ingest both Actors, then search with citations.
# Needs APIFY_TOKEN in the environment. Creates .venv on first use.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${APIFY_TOKEN:-}" ]]; then
  echo "APIFY_TOKEN is not set. See README.md, Quick start." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating .venv and installing python/requirements.txt ..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r python/requirements.txt
fi

.venv/bin/python python/basic_pipeline.py
echo
.venv/bin/python python/search_local.py "${1:-What affects the reserve balances of depository institutions?}"
