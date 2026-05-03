#!/usr/bin/env bash
set -e

DATA_DIR="${DATA_ROOT:-/data/survey}"
RATINGS_DIR="$(dirname "${RATINGS_PATH:-/data/ratings/ratings.jsonl}")"

mkdir -p "$DATA_DIR" "$RATINGS_DIR"

echo "=== Downloading survey data from HF Dataset ==="
python3 - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download

repo = os.environ.get("HF_DATASET_REPO", "shaluoyan523/guiguard-bench-survey-data")
dest = os.environ.get("DATA_ROOT", "/data/survey")

print(f"Downloading {repo} -> {dest} (JSON only, images served via HF URLs)")
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=dest,
    ignore_patterns=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"],
)
print("Download complete.")
PYEOF

echo "=== Starting survey server ==="
exec python3 server.py --host 0.0.0.0 --port 7860
