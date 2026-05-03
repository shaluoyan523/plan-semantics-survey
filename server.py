#!/usr/bin/env python3
"""Local survey server for protected planner semantic ratings."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import re
import secrets
import sys
import time
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path configuration — override via environment variables for HF Spaces.
# DATA_ROOT: directory that contains GUIGaurd-Bench-master/runs, PC, Android, …
# HF_DATASET_REPO: if set, images are served as HF Hub URLs instead of locally.
# RATINGS_PATH: where to store ratings.jsonl (default: next to this file).
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("DATA_ROOT", "/vepfs-mlp2/project-infoengine/guanhaoxiang/yanxiwang"))
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")  # e.g. "shaluoyan523/guiguard-bench-survey-data"
# RUNS_DIR: directory containing owl_pc_gt, owl_pc_step_*, etc.
# Defaults to ROOT/GUIGaurd-Bench-master/runs for local use.
# In Docker/HF Spaces, set to /app/data/runs.
RUNS = Path(os.environ.get("RUNS_DIR", str(ROOT / "GUIGaurd-Bench-master" / "runs")))
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
_ratings_default = APP_DIR / "ratings.jsonl"
RATINGS_PATH = Path(os.environ.get("RATINGS_PATH", str(_ratings_default)))

METHODS = ("black", "mosaic", "randblocks", "replace_llm")
SCORE_VALUES = (0, 1, 2, 3, 4)
DIFF_VALUES = tuple(range(-4, 5))
INITIAL_CONFUSION = [
    [6, 0, 0, 0, 0],
    [12, 36, 12, 1, 0],
    [5, 52, 218, 53, 7],
    [1, 5, 21, 90, 20],
    [0, 0, 0, 4, 44],
]

RUNS_CONFIG: dict[str, list[dict[str, Any]]] = {
    "pc": [
        {"run_id": "step_1_10pct", "label": "10% steps",  "dir": RUNS / "owl_pc_step_1_10pct", "ladder_step": "step_1_10pct"},
        {"run_id": "step_2_20pct", "label": "20% steps",  "dir": RUNS / "owl_pc_step_2_20pct", "ladder_step": "step_2_20pct"},
        {"run_id": "step_3_30pct", "label": "30% steps",  "dir": RUNS / "owl_pc_step_3_30pct", "ladder_step": "step_3_30pct"},
        {"run_id": "step_4_70pct", "label": "70% steps",  "dir": RUNS / "owl_pc_step_4_70pct", "ladder_step": "step_4_70pct"},
        {"run_id": "step_5_80pct", "label": "80% steps",  "dir": RUNS / "owl_pc_step_5_80pct", "ladder_step": "step_5_80pct"},
        {"run_id": "full",         "label": "Full (100%)","dir": RUNS / "owl_pc",              "ladder_step": "step_6_100pct"},
    ],
    "android": [
        {"run_id": "step_1_10pct", "label": "10% steps",  "dir": RUNS / "owl_android_step_1_10pct", "ladder_step": "step_1_10pct"},
        {"run_id": "step_2_20pct", "label": "20% steps",  "dir": RUNS / "owl_android_step_2_20pct", "ladder_step": "step_2_20pct"},
        {"run_id": "step_3_30pct", "label": "30% steps",  "dir": RUNS / "owl_android_step_3_30pct", "ladder_step": "step_3_30pct"},
        {"run_id": "step_4_40pct", "label": "40% steps",  "dir": RUNS / "owl_android_step_4_40pct", "ladder_step": "step_4_40pct"},
        {"run_id": "step_5_50pct", "label": "50% steps",  "dir": RUNS / "owl_android_step_5_50pct", "ladder_step": "step_5_50pct"},
        {"run_id": "step_60pct",   "label": "60% steps",  "dir": RUNS / "owl_android_step_60pct",   "ladder_step": "step_60pct"},
        {"run_id": "step_70pct",   "label": "70% steps",  "dir": RUNS / "owl_android_step_70pct",   "ladder_step": "step_70pct"},
        {"run_id": "full",         "label": "Full (100%)","dir": RUNS / "owl_android",              "ladder_step": "step_6_100pct"},
    ],
}

GT_DIRS = {
    "pc": RUNS / "owl_pc_gt" / "gt_results",
    "android": RUNS / "owl_android_gt" / "gt_results",
}
GT_SUFFIX = "_gt_result.json"
RESULT_SUFFIX = "_result.json"

PUBLIC_DIRS = {
    "pc": ROOT / "PC",
    "android": ROOT / "Android",
}
LADDER_DIRS = {
    "pc": ROOT / "PC_public_ladder",
    "android": ROOT / "Android_public_ladder",
}

DEFAULT_RUN = {"pc": "step_5_80pct", "android": "step_70pct"}

JUDGE_EVAL_DIRS: dict[tuple[str, str], Path] = {
    ("pc",      "step_1_10pct"): RUNS / "owl_pc_step_1_10pct" / "judge_evaluation",
    ("pc",      "step_2_20pct"): RUNS / "owl_pc_step_2_20pct" / "judge_evaluation",
    ("pc",      "step_3_30pct"): RUNS / "owl_pc_step_3_30pct" / "judge_evaluation",
    ("pc",      "step_4_70pct"): RUNS / "owl_pc_step_4_70pct" / "judge_evaluation",
    ("pc",      "step_5_80pct"): RUNS / "owl_pc_step_5_80pct" / "judge_evaluation",
    ("pc",      "full"):         RUNS / "owl_pc"               / "judge_evaluation",
    ("android", "step_1_10pct"): RUNS / "owl_android_step_1_10pct" / "judge_evaluation",
    ("android", "step_2_20pct"): RUNS / "owl_android_step_2_20pct" / "judge_evaluation",
    ("android", "step_3_30pct"): RUNS / "owl_android_step_3_30pct" / "judge_evaluation",
    ("android", "step_4_40pct"): RUNS / "owl_android_step_4_40pct" / "judge_evaluation",
    ("android", "step_5_50pct"): RUNS / "owl_android_step_5_50pct" / "judge_evaluation",
    ("android", "step_60pct"):   RUNS / "owl_android_step_60pct"   / "judge_evaluation",
    ("android", "step_70pct"):   RUNS / "owl_android_step_70pct"   / "judge_evaluation",
    ("android", "full"):         RUNS / "owl_android"               / "judge_evaluation",
}


@dataclass(frozen=True)
class TaskRef:
    dataset: str
    run_id: str
    task_id: str
    gt_file: Path
    protected_files: dict[str, Path]
    instruction: str
    total_steps: int
    has_public_images: bool


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_task_id(path: Path, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        return path.stem
    return name[: -len(suffix)]


def candidate_public_names(dataset: str, task_id: str, gt_data: dict[str, Any]) -> list[str]:
    candidates = [task_id]
    for action in gt_data.get("actions") or []:
        screenshot_path = action.get("screenshot_path")
        if not screenshot_path:
            continue
        path = Path(screenshot_path)
        if dataset == "android" and path.parent.name == "images":
            candidates.append(path.parent.parent.name)
        else:
            candidates.append(path.parent.name)
    seen: set[str] = set()
    return [item for item in candidates if item and not (item in seen or seen.add(item))]


def sort_key_for_step(path: Path) -> tuple[int, str]:
    match = re.search(r"(?:^|_)step_(\d+)|screenshot_[^/]*-(\d+)-", path.name)
    if match:
        number = next(int(g) for g in match.groups() if g is not None)
        return (number, path.name)
    return (10**9, path.name)


def _path_to_hf_url(path: Path) -> str:
    """Convert a local absolute path to a HF Dataset resolve URL."""
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        rel = path
    return f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{rel}"


def image_paths_from_actions(actions: list[dict[str, Any]]) -> list[Path]:
    """Return screenshot paths from action list.
    In HF mode, paths are returned even if they don't exist locally (URLs will be constructed later).
    """
    paths: list[Path] = []
    for action in actions:
        sp = action.get("screenshot_path")
        if sp:
            path = Path(sp)
            if HF_DATASET_REPO:
                paths.append(path)  # trust the path, URL generated at serve time
            elif path.exists() and path.is_file():
                paths.append(path)
    return paths


def fallback_public_images(dataset: str, task_id: str, gt_data: dict[str, Any]) -> list[Path]:
    if HF_DATASET_REPO:
        return []  # HF mode: screenshot_path in actions is used; no local glob
    public_dir = PUBLIC_DIRS[dataset]
    for name in candidate_public_names(dataset, task_id, gt_data):
        public_task = public_dir / name
        if not public_task.exists():
            continue
        if dataset == "android":
            images = sorted((public_task / "images").glob("*.png"), key=sort_key_for_step)
        else:
            images = sorted(public_task.glob("step_*.png"), key=sort_key_for_step)
        if images:
            return images
    return []


def protected_ladder_images(
    dataset: str, task_id: str, run_id: str, method: str, protected_data: dict[str, Any]
) -> list[Path]:
    if HF_DATASET_REPO:
        return []  # skip ladder images in HF mode; use action screenshot_path only
    paths = image_paths_from_actions(protected_data.get("actions") or [])
    if paths:
        return paths
    ladder_dir = LADDER_DIRS[dataset]
    ladder_task = ladder_dir / task_id
    if not ladder_task.exists():
        return []
    ladder_step = next((rc["ladder_step"] for rc in RUNS_CONFIG.get(dataset, []) if rc["run_id"] == run_id), None)
    if ladder_step:
        preferred = ladder_task / ladder_step / method
        if preferred.exists():
            images = sorted(preferred.glob("*.png"), key=sort_key_for_step)
            if images:
                return images
    for step_dir in sorted(
        [d for d in ladder_task.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: sort_key_for_step(d),
    ):
        method_dir = step_dir / method
        if method_dir.exists():
            images = sorted(method_dir.glob("*.png"), key=sort_key_for_step)
            if images:
                return images
    return []


def normalize_action(action: Any) -> dict[str, Any]:
    return action if isinstance(action, dict) else {}


def load_tasks() -> list[TaskRef]:
    tasks: list[TaskRef] = []
    for dataset, run_cfgs in RUNS_CONFIG.items():
        gt_dir = GT_DIRS.get(dataset)
        if not gt_dir or not gt_dir.exists():
            continue
        gt_cache: dict[str, tuple[str, int, bool, Path]] = {}
        for gt_file in sorted(gt_dir.glob(f"*{GT_SUFFIX}")):
            task_id = extract_task_id(gt_file, GT_SUFFIX)
            try:
                gt_data = read_json(gt_file)
            except (OSError, json.JSONDecodeError):
                continue
            instruction = str(gt_data.get("instruction") or task_id)
            actions = [normalize_action(a) for a in (gt_data.get("actions") or [])]
            has_public = bool(
                image_paths_from_actions(actions) or fallback_public_images(dataset, task_id, gt_data)
            )
            gt_cache[task_id] = (instruction, max(1, len(actions)), has_public, gt_file)

        for rc in run_cfgs:
            run_id = rc["run_id"]
            protected_base = rc["dir"] / "planner_results"
            if not protected_base.exists():
                continue
            for task_id, (instruction, total_steps, has_public, gt_file) in gt_cache.items():
                existing = {
                    m: protected_base / m / f"{task_id}{RESULT_SUFFIX}"
                    for m in METHODS
                    if (protected_base / m / f"{task_id}{RESULT_SUFFIX}").exists()
                }
                if not existing:
                    continue
                tasks.append(TaskRef(
                    dataset=dataset, run_id=run_id, task_id=task_id,
                    gt_file=gt_file, protected_files=existing,
                    instruction=instruction, total_steps=total_steps,
                    has_public_images=has_public,
                ))

    run_rank = {(ds, rc["run_id"]): i for ds, cfgs in RUNS_CONFIG.items() for i, rc in enumerate(cfgs)}

    def sort_key(t: TaskRef) -> tuple[int, int, str]:
        return (0 if t.dataset == "pc" else 1, run_rank.get((t.dataset, t.run_id), 99), t.task_id)

    return sorted(tasks, key=sort_key)


TASKS = load_tasks()
TASK_INDEX: dict[tuple[str, str, str], TaskRef] = {
    (t.dataset, t.run_id, t.task_id): t for t in TASKS
}
IMAGE_INDEX: dict[str, Path] = {}
SAMPLE_INDEX: dict[str, tuple[str, str, str, str, int]] = {}


def coerce_score(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return score if score in SCORE_VALUES else None


def load_llm_scores() -> dict[tuple[str, str, str, str, int], int]:
    scores: dict[tuple[str, str, str, str, int], int] = {}
    for (dataset, run_id), directory in JUDGE_EVAL_DIRS.items():
        for method in METHODS:
            path = directory / f"{method}_evaluation.json"
            if not path.exists():
                continue
            try:
                data = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            for result in data.get("results") or []:
                task_name = str(result.get("task_name") or "")
                for step_result in result.get("step_results") or []:
                    step = int(step_result.get("step") or 0)
                    score = coerce_score(step_result.get("score"))
                    if task_name and step > 0 and score is not None:
                        scores.setdefault((dataset, run_id, task_name, method, step), score)
    return scores


LLM_SCORE_INDEX = load_llm_scores()


def make_image_url(path: Path) -> str:
    """Return a URL for a screenshot path — HF Hub URL in HF mode, local API URL otherwise."""
    if HF_DATASET_REPO:
        return _path_to_hf_url(path)
    key = secrets.token_urlsafe(16)
    IMAGE_INDEX[key] = path.resolve()
    return f"/api/image/{key}"


def sanitize_public_text(value: Any) -> Any:
    if isinstance(value, str):
        text = re.sub(r"/[^\s`'\"),]+", "[path]", value)
        text = re.sub(r"\b[\w.-]+(?:_result|_gt_result)\.json\b", "[file]", text)
        text = re.sub(r"\b[\w@.-]+\.(?:png|jpg|jpeg|jsonl|json|txt|log)\b", "[file]", text, flags=re.IGNORECASE)
        text = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "[sample id]", text)
        text = re.sub(r"\b20\d{6}_[^\s`'\"),]+", "[sample id]", text)
        return text
    if isinstance(value, list):
        return [sanitize_public_text(i) for i in value]
    if isinstance(value, dict):
        return {k: sanitize_public_text(v) for k, v in value.items()}
    return value


def new_sample_token(dataset: str, run_id: str, task_id: str, method: str, step: int) -> str:
    token = secrets.token_urlsafe(16)
    SAMPLE_INDEX[token] = (dataset, run_id, task_id, method, step)
    return token


def available_methods(dataset: str) -> list[str]:
    return sorted({m for t in TASKS if t.dataset == dataset for m in t.protected_files})


def random_target_payload(dataset: str, method: str | None, avoid_token: str | None = None) -> dict[str, Any]:
    if dataset not in RUNS_CONFIG:
        raise ValueError("Unknown dataset.")
    all_methods = available_methods(dataset)
    if method not in all_methods:
        method = all_methods[-1] if all_methods else METHODS[-1]
    candidates: list[tuple[TaskRef, int]] = []
    covered: list[tuple[TaskRef, int]] = []
    for task in TASKS:
        if task.dataset != dataset or method not in task.protected_files:
            continue
        for step in range(1, max(1, task.total_steps) + 1):
            candidates.append((task, step))
            if (dataset, task.run_id, task.task_id, method, step) in LLM_SCORE_INDEX:
                covered.append((task, step))
    if covered:
        candidates = covered
    if not candidates:
        raise ValueError("No samples available for this dataset and method.")
    avoid = SAMPLE_INDEX.get(avoid_token or "")
    if avoid and len(candidates) > 1:
        av = avoid
        candidates = [
            (t, s) for t, s in candidates
            if not (t.dataset == av[0] and t.run_id == av[1] and t.task_id == av[2] and method == av[3] and s == av[4])
        ]
    task, step = random.choice(candidates)
    return task_payload(task, method, step, public=True)


def action_for_step(actions: list[dict[str, Any]], step: int) -> dict[str, Any]:
    if not actions:
        return {}
    for action in actions:
        if action.get("step") == step:
            return action
    return actions[max(0, min(step - 1, len(actions) - 1))]


def task_payload(task: TaskRef, method: str | None, step: int | None, public: bool = False) -> dict[str, Any]:
    dataset, run_id, task_id = task.dataset, task.run_id, task.task_id
    if method not in task.protected_files:
        method = sorted(task.protected_files)[0]

    gt_data = read_json(task.gt_file)
    protected_data = read_json(task.protected_files[method])
    before_actions = [normalize_action(a) for a in (gt_data.get("actions") or [])]
    after_actions  = [normalize_action(a) for a in (protected_data.get("actions") or [])]
    total_steps = max(1, min(len(before_actions) or 1, len(after_actions) or len(before_actions) or 1))
    current_step = max(1, min(step or 1, total_steps))

    before_action = action_for_step(before_actions, current_step)
    after_action  = action_for_step(after_actions,  current_step)

    source_images = image_paths_from_actions(before_actions) or fallback_public_images(dataset, task_id, gt_data)
    protected_images = protected_ladder_images(dataset, task_id, run_id, method, protected_data)
    trajectory = source_images if source_images else protected_images
    image_variant = "public" if source_images else "protected"

    visible = trajectory[:current_step]
    images = [
        {"url": make_image_url(p), "label": f"Screenshot {i + 1}", "step": i + 1}
        for i, p in enumerate(visible)
    ]
    current_image_index = max(0, min(current_step - 1, len(images) - 1)) if images else 0

    run_label = next((rc["label"] for rc in RUNS_CONFIG.get(dataset, []) if rc["run_id"] == run_id), run_id)
    payload = {
        "dataset": dataset,
        "run": run_id,
        "runLabel": run_label,
        "sampleLabel": f"{dataset.upper()} random target",
        "instruction": sanitize_public_text(task.instruction),
        "method": method,
        "methods": available_methods(dataset),
        "step": current_step,
        "totalSteps": total_steps,
        "imageVariant": image_variant,
        "images": images,
        "currentImageIndex": current_image_index,
        "beforePlan": sanitize_public_text(before_action.get("plan") or ""),
        "afterPlan":  sanitize_public_text(after_action.get("plan") or ""),
        "beforeAction": sanitize_public_text(before_action.get("actions") or []),
        "afterAction":  sanitize_public_text(after_action.get("actions") or []),
        "meta": {
            "beforeCompleted": bool(gt_data.get("completed")),
            "afterCompleted": bool(protected_data.get("completed")),
        },
    }
    if public:
        payload["sampleToken"] = new_sample_token(dataset, run_id, task_id, method, current_step)
    else:
        payload["taskId"] = task_id
    return payload


def catalog_payload() -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in RUNS_CONFIG:
        datasets[dataset] = {
            "count": sum(1 for t in TASKS if t.dataset == dataset),
            "methods": available_methods(dataset),
        }
    default_ds = "pc" if datasets.get("pc", {}).get("count") else next(iter(datasets), "pc")
    return {"datasets": datasets, "default": {"dataset": default_ds, "method": "replace_llm"}, "counts": {ds: d["count"] for ds, d in datasets.items()}}


def empty_confusion() -> list[list[int]]:
    return [[0] * len(SCORE_VALUES) for _ in SCORE_VALUES]


def difference_counts(confusion: list[list[int]]) -> dict[str, int]:
    counts = {str(d): 0 for d in DIFF_VALUES}
    for h, row in enumerate(confusion):
        for l, n in enumerate(row):
            counts[str(l - h)] += n
    return counts


def summarize_confusion(confusion: list[list[int]]) -> dict[str, Any]:
    total = sum(sum(row) for row in confusion)
    same = sum(confusion[i][i] for i in SCORE_VALUES)
    pm1 = sum(confusion[h][l] for h, row in enumerate(confusion) for l, _ in enumerate(row) if abs(l - h) == 1)
    return {
        "total": total,
        "same": same,
        "samePct": (same / total * 100) if total else 0,
        "plusMinus1": pm1,
        "plusMinus1Pct": (pm1 / total * 100) if total else 0,
    }


def user_entries() -> list[dict[str, Any]]:
    if not RATINGS_PATH.exists():
        return []
    entries = []
    with RATINGS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            human = coerce_score(entry.get("score"))
            step = int(entry.get("step") or 0)
            dataset = str(entry.get("dataset") or "")
            task_id = str(entry.get("taskId") or "")
            method = str(entry.get("method") or "")
            run_id = str(entry.get("runId") or DEFAULT_RUN.get(dataset, ""))
            llm = LLM_SCORE_INDEX.get((dataset, run_id, task_id, method, step))
            if human is None or llm is None:
                continue
            entries.append({"dataset": dataset, "run": run_id, "method": method, "step": step, "humanScore": human, "llmScore": llm})
    return entries


def results_payload() -> dict[str, Any]:
    user_confusion = empty_confusion()
    by_method: dict[str, int] = {m: 0 for m in METHODS}
    by_dataset: dict[str, int] = {ds: 0 for ds in RUNS_CONFIG}
    for e in user_entries():
        user_confusion[e["humanScore"]][e["llmScore"]] += 1
        by_method[e["method"]] = by_method.get(e["method"], 0) + 1
        by_dataset[e["dataset"]] = by_dataset.get(e["dataset"], 0) + 1
    combined = [
        [INITIAL_CONFUSION[r][c] + user_confusion[r][c] for c in SCORE_VALUES]
        for r in SCORE_VALUES
    ]
    return {
        "scores": list(SCORE_VALUES),
        "diffs": list(DIFF_VALUES),
        "initial": {"confusion": INITIAL_CONFUSION, "differenceCounts": difference_counts(INITIAL_CONFUSION), "summary": summarize_confusion(INITIAL_CONFUSION)},
        "user": {"confusion": user_confusion, "differenceCounts": difference_counts(user_confusion), "summary": summarize_confusion(user_confusion), "byMethod": by_method, "byDataset": by_dataset},
        "combined": {"confusion": combined, "differenceCounts": difference_counts(combined), "summary": summarize_confusion(combined)},
        "llmScoreCoverage": len(LLM_SCORE_INDEX),
    }


def write_rating(payload: dict[str, Any], client: str) -> dict[str, Any]:
    score = payload.get("score")
    if score not in (0, 1, 2, 3, 4):
        raise ValueError("Score must be 0–4.")
    token = str(payload.get("sampleToken") or "")
    sample = SAMPLE_INDEX.get(token)
    if sample is None:
        raise ValueError("Unknown sample.")
    dataset, run_id, task_id, method, step = sample
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "client": client, "dataset": dataset, "runId": run_id,
        "taskId": task_id, "method": method, "step": step,
        "score": score, "note": str(payload.get("note") or ""),
    }
    RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RATINGS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"ok": True, "saved": {"dataset": dataset, "run": run_id, "method": method, "step": step, "score": score}}


class SurveyHandler(BaseHTTPRequestHandler):
    server_version = "PlanSemanticsSurvey/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.client_address[0]} - [{self.log_date_time_string()}] {fmt % args}\n")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: HTTPStatus) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self.serve_static(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                self.serve_static(STATIC_DIR / path.removeprefix("/static/"))
            elif path == "/api/catalog":
                self.send_json(catalog_payload())
            elif path == "/api/results":
                self.send_json(results_payload())
            elif path == "/api/random-task":
                dataset = query.get("dataset", ["pc"])[0]
                method  = query.get("method",  [None])[0]
                avoid   = query.get("avoid",   [None])[0]
                self.send_json(random_target_payload(dataset, method, avoid))
            elif path == "/api/task":
                token = query.get("sample", [""])[0]
                sample = SAMPLE_INDEX.get(token)
                if sample is None:
                    raise ValueError("Unknown sample.")
                dataset, run_id, task_id, method, step = sample
                task = TASK_INDEX.get((dataset, run_id, task_id))
                if task is None:
                    raise ValueError("Unknown sample.")
                self.send_json(task_payload(task, method, step, public=True))
            elif path.startswith("/api/image/") and not HF_DATASET_REPO:
                key = path.rsplit("/", 1)[-1]
                image_path = IMAGE_INDEX.get(key)
                if image_path is None:
                    self.send_error_json("Image not found.", HTTPStatus.NOT_FOUND)
                else:
                    self.serve_file(image_path, allow_any=True)
            else:
                self.send_error_json("Not found.", HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError) as exc:
            self.send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_error_json(f"Server error: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        if urllib.parse.urlparse(self.path).path != "/api/rating":
            self.send_error_json("Not found.", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json(write_rating(payload, self.client_address[0]))
        except (json.JSONDecodeError, ValueError) as exc:
            self.send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_error_json(f"Server error: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_static(self, path: Path) -> None:
        resolved = path.resolve()
        if STATIC_DIR.resolve() not in [resolved, *resolved.parents]:
            self.send_error_json("Invalid path.", HTTPStatus.BAD_REQUEST)
            return
        self.serve_file(resolved)

    def serve_file(self, path: Path, allow_any: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self.send_error_json("File not found.", HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "max-age=60")
        self.end_headers()
        with path.open("rb") as f:
            while chunk := f.read(131072):
                self.wfile.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not TASKS:
        raise SystemExit("No paired tasks found. Check DATA_ROOT.")
    server = ThreadingHTTPServer((args.host, args.port), SurveyHandler)
    hf_note = f" | HF images from {HF_DATASET_REPO}" if HF_DATASET_REPO else ""
    print(f"Serving {len(TASKS)} tasks ({len(LLM_SCORE_INDEX)} LLM scores) at http://{args.host}:{args.port}{hf_note}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
