from __future__ import annotations

import json
import os
import platform
import random
import resource
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import save_tensors_deterministic
from .backends import LayaBackend
from .io import load_suite, write_json
from .metrics import (
    classification_metrics,
    option_order_instability,
    paired_summary,
    paraphrase_consistency,
)
from .schemas import Example, TaskSpec
from .store import ExperimentStore, now_iso
from .strategies import FittedStrategy, StrategyNotApplicable, create_strategy


def benchmark_suite(
    suite_path: Path,
    methods: list[str],
    backend: LayaBackend,
    store: ExperimentStore,
    artifacts_root: Path,
    seed: int = 0,
    batch_size: int = 64,
    option_permutations: int = 3,
) -> str:
    _manifest, tasks, splits = load_suite(suite_path)
    run_id = datetime_run_id()
    run_root = artifacts_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    store.create_run(
        {
            "id": run_id,
            "created_at": now_iso(),
            "status": "running",
            "suite_path": str(suite_path.resolve()),
            "base_model": backend.info.model_id,
            "base_revision": backend.info.revision,
            "backend": backend.info.name,
            "seed": seed,
            "environment_json": json.dumps(environment_info(), sort_keys=True),
            "git_commit": git_commit(),
        }
    )
    try:
        for task in tasks:
            task_rows = {
                name: [x for x in rows if x.task_name == task.name] for name, rows in splits.items()
            }
            if not task_rows["specialization"] or not task_rows["hidden"]:
                raise ValueError(f"task {task.name} lacks specialization or hidden examples")
            baseline_predictions: dict[str, str] = {}
            for method in methods:
                strategy = create_strategy(method)
                try:
                    fitted = strategy.fit(
                        task, task_rows["specialization"], task_rows["validation"], backend
                    )
                except (NotImplementedError, StrategyNotApplicable) as exc:
                    write_json(run_root / task.name / f"{method}.skip.json", {"reason": str(exc)})
                    continue
                _save_fitted(run_root / task.name / method, task, fitted, backend)
                store.add_specialization(
                    f"{run_id}:{task.name}:{fitted.artifact.strategy}",
                    run_id,
                    task.name,
                    fitted.artifact.strategy,
                    str((run_root / task.name / method).resolve()),
                )
                result, prediction_rows = evaluate_fitted(
                    fitted, task, task_rows, backend, run_id, seed, batch_size, option_permutations
                )
                if fitted.artifact.strategy == "baseline":
                    baseline_predictions = {
                        x["example_id"]: x["predicted"] for x in prediction_rows
                    }
                for row in prediction_rows:
                    row["baseline_predicted"] = baseline_predictions.get(row["example_id"])
                store.add_result(
                    {"run_id": run_id, "task_name": task.name, "domain": task.domain, **result}
                )
                store.add_predictions(prediction_rows)
        store.set_status(run_id, "completed")
    except BaseException:
        store.set_status(run_id, "failed")
        raise
    return run_id


def evaluate_fitted(
    fitted: FittedStrategy,
    task: TaskSpec,
    rows: dict[str, list[Example]],
    backend: LayaBackend,
    run_id: str,
    seed: int,
    batch_size: int,
    option_permutations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation_rows = rows["validation"] + rows["hidden"] + rows["paraphrase"] + rows["hard"]
    before_mem = _peak_memory_mb()
    started = time.perf_counter()
    predictions = []
    batch_latencies = []
    for start in range(0, len(evaluation_rows), batch_size):
        batch = evaluation_rows[start : start + batch_size]
        batch_start = time.perf_counter()
        predictions.extend(fitted.predict([x.input for x in batch]))
        batch_latencies.append(time.perf_counter() - batch_start)
    elapsed = time.perf_counter() - started
    expected = [x.label for x in evaluation_rows]
    probabilities = [x["probabilities"] for x in predictions]
    overall = classification_metrics(expected, probabilities, task.decision.labels)
    per_split = {}
    cursor = 0
    for split in ("validation", "hidden", "paraphrase", "hard"):
        count = len(rows[split])
        if count:
            per_split[split] = classification_metrics(
                [x.label for x in rows[split]],
                probabilities[cursor : cursor + count],
                task.decision.labels,
            )
        cursor += count
    rng = random.Random(seed)
    option_runs: list[list[str]] = []
    if fitted.artifact.decision_component == "laya_head" and fitted.artifact.strategy == "baseline":
        hidden_states = [x.input for x in rows["hidden"]]
        for _ in range(option_permutations):
            permuted = task.model_copy(deep=True)
            rng.shuffle(permuted.decision.labels)
            option_runs.append([x["label"] for x in backend.predict_batch(hidden_states, permuted)])
    else:
        hidden_start = len(rows["validation"])
        hidden_end = hidden_start + len(rows["hidden"])
        option_runs = [[x["label"] for x in predictions[hidden_start:hidden_end]]] * max(
            option_permutations, 2
        )
    noise_rows = rows["hidden"]
    noisy = [
        x.input + "\nIrrelevant note: calendar color is blue; reference 8472." for x in noise_rows
    ]
    noisy_predictions = fitted.predict(noisy) if noisy else []
    noisy_accuracy = (
        float(np.mean([p["label"] == e.label for p, e in zip(noisy_predictions, noise_rows)]))
        if noisy
        else None
    )
    hidden_accuracy = per_split.get("hidden", {}).get("accuracy")
    style_accuracy = _style_accuracy(evaluation_rows, predictions)
    robustness = {
        "option_order_instability": option_order_instability(option_runs),
        "paraphrase_consistency": paraphrase_consistency(
            [x["label"] for x in predictions], [x.pair_id for x in evaluation_rows]
        ),
        "noise_accuracy": noisy_accuracy,
        "noise_accuracy_delta": noisy_accuracy - hidden_accuracy
        if noisy_accuracy is not None and hidden_accuracy is not None
        else None,
        "style_accuracy": style_accuracy,
        "style_accuracy_spread": (
            max(style_accuracy.values()) - min(style_accuracy.values())
            if len(style_accuracy) > 1
            else None
        ),
    }
    timings = {
        "specialization_seconds": fitted.specialization_seconds,
        "warm_total_seconds": elapsed,
        "latency_per_sample_ms": 1000 * elapsed / len(evaluation_rows),
        "batch_latency_mean_ms": 1000 * float(np.mean(batch_latencies)),
        "throughput_samples_per_second": len(evaluation_rows) / elapsed if elapsed else None,
        "cold_load_seconds": getattr(backend, "cold_load_seconds", None),
        "peak_memory_mb": max(_peak_memory_mb() - before_mem, 0.0),
        "model_calls": len(evaluation_rows)
        if fitted.artifact.decision_component in {"laya_head", "pairwise_laya_head"}
        else len(batch_latencies),
    }
    result = {
        "strategy": fitted.artifact.strategy,
        "decision_component": fitted.artifact.decision_component,
        "strategy_params": fitted.artifact.params,
        "metrics": {"overall": overall, "splits": per_split, "robustness": robustness},
        "timings": timings,
    }
    stored = [
        {
            "run_id": run_id,
            "task_name": task.name,
            "strategy": fitted.artifact.strategy,
            "example_id": example.id,
            "split": example.split,
            "input": example.input,
            "expected": example.label,
            "predicted": prediction["label"],
            "probabilities": prediction["probabilities"],
        }
        for example, prediction in zip(evaluation_rows, predictions)
    ]
    return result, stored


def aggregate_run(
    data: dict[str, Any],
    metric: str = "accuracy",
    bootstrap_samples: int = 0,
    holdout_domain: str | None = None,
) -> dict[str, Any]:
    by_strategy: dict[str, dict[str, float]] = {}
    for row in data["results"]:
        if holdout_domain is not None and row["domain"] != holdout_domain:
            continue
        by_strategy.setdefault(row["strategy"], {})[row["task_name"]] = row["metrics"]["overall"][
            metric
        ]
    baseline = by_strategy.get("baseline")
    if not baseline:
        raise ValueError("run has no baseline results")
    return {
        strategy: paired_summary(scores, baseline, bootstrap_samples=bootstrap_samples)
        for strategy, scores in sorted(by_strategy.items())
    }


def _save_fitted(path: Path, task: TaskSpec, fitted: FittedStrategy, backend: LayaBackend) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(
        path / "specialization.json",
        {
            "task": task.model_dump(mode="json"),
            "artifact": fitted.artifact.model_dump(mode="json"),
            "checkpoint": {
                "backend": backend.info.name,
                "model_id": backend.info.model_id,
                "revision": backend.info.revision,
            },
        },
    )
    save_tensors_deterministic(
        path / "vectors.safetensors",
        {key: np.ascontiguousarray(value) for key, value in sorted(fitted.arrays.items())},
    )


def _style_accuracy(rows: list[Example], predictions: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = {}
    for row, prediction in zip(rows, predictions):
        buckets.setdefault(row.source_style or "unknown", []).append(
            row.label == prediction["label"]
        )
    return {key: float(np.mean(values)) for key, values in sorted(buckets.items())}


def _peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024)


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pid": os.getpid(),
    }


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def datetime_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
