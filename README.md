---
title: Planner Semantics Survey
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Planner Semantics Survey

Human evaluation tool for assessing semantic consistency of privacy-protected GUI agent plans.

Evaluators compare **before** and **after** planner outputs (before/after applying a privacy protection method) and rate their semantic consistency on a 0–4 scale.

## Running locally

```bash
pip install -r requirements.txt
DATA_ROOT=/path/to/data python server.py --port 7860
```

## Environment variables

| Variable | Description |
|---|---|
| `DATA_ROOT` | Root directory containing `GUIGaurd-Bench-master/runs`, `PC/`, `Android/` |
| `HF_DATASET_REPO` | HF Dataset repo ID for image hosting (e.g. `shaluoyan523/guiguard-bench-survey-data`) |
| `RATINGS_PATH` | Path to persist ratings JSONL file |
