from __future__ import annotations

import json
import os
import re
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from .artifact import export_specialization, load_saved_fitted
from .backends import LayaBackend, create_backend
from .evaluation import benchmark_suite
from .generation import (
    generate_benchmark_splits,
    generate_specialization,
    infer_task,
    remove_generated_leakage,
)
from .io import load_suite, save_suite
from .models import model_catalog
from .providers import OpenAICompatibleProvider
from .schemas import Example, SuiteManifest, TaskSpec
from .store import ExperimentStore


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(ApiModel):
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = Field(min_length=1)
    api_key: str = ""


class SettingsRequest(ApiModel):
    provider: ProviderConfig


class PlanRequest(ApiModel):
    description: str = Field(min_length=10)
    labels: list[str] | None = None
    provider: ProviderConfig
    seed: int = 0


class GenerateRequest(ApiModel):
    task: TaskSpec
    specialization_provider: ProviderConfig
    benchmark_provider: ProviderConfig
    specialization_examples: int = Field(default=30, ge=6, le=500)
    validation_examples: int = Field(default=18, ge=4, le=500)
    hidden_examples: int = Field(default=40, ge=4, le=2000)
    paraphrase_examples: int = Field(default=16, ge=4, le=500)
    hard_examples: int = Field(default=16, ge=4, le=500)
    seed: int = 0


class RunRequest(ApiModel):
    name: str | None = Field(default=None, max_length=120)
    task: TaskSpec
    examples: list[dict[str, Any]]
    methods: list[str] = Field(min_length=1)
    backend: str = "mlx"
    model: str = "aac6fef/laya-mlx"
    revision: str | None = None
    device: str | None = None
    seed: int = 0


class NameRequest(ApiModel):
    name: str = Field(min_length=1, max_length=120)


class KeepRequest(ApiModel):
    task: str
    strategy: str
    kept: bool


class DefaultRequest(ApiModel):
    task: str
    strategy: str


class PlaygroundRequest(ApiModel):
    task: str
    strategy: str
    text: str = Field(min_length=1, max_length=20_000)


class ExportResultRequest(ApiModel):
    task: str
    strategy: str
    name: str | None = Field(default=None, max_length=120)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: int = 0
    phase: str = "Preparing"
    message: str = "The job will start shortly."
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "phase": self.phase,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self, workers: int = 2):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="laya-studio")

    def start(self, kind: str, work: Callable[[Callable[..., None]], dict[str, Any]]) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind)
        with self.lock:
            self.jobs[job.id] = job

        def update(progress: int, phase: str, message: str) -> None:
            with self.lock:
                job.progress = max(0, min(100, progress))
                job.phase = phase
                job.message = message
                job.updated_at = datetime.now(timezone.utc).isoformat()

        def execute() -> None:
            with self.lock:
                job.status = "running"
            try:
                result = work(update)
                with self.lock:
                    job.result = result
                    job.status = "completed"
                    job.progress = 100
                    job.phase = "Completed"
                    job.message = "Everything is ready."
            except BaseException as exc:
                with self.lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.message = "The job failed. Check the settings and try again."
                    # Full trace stays server-side; generated content is never executed.
                    traceback.print_exc()
            finally:
                job.updated_at = datetime.now(timezone.utc).isoformat()

        self.pool.submit(execute)
        return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]


class StudioService:
    ENV_KEYS = (
        "LAYA_LAB_LLM_BASE_URL",
        "LAYA_LAB_LLM_API_KEY",
        "LAYA_LAB_SPECIALIZATION_MODEL",
    )

    def __init__(self, root: Path, database: Path, env_path: Path | None = None):
        self.root = root
        self.drafts = root / "drafts"
        self.exports = root / "exports"
        self.runs = root / "runs"
        for path in (self.drafts, self.exports, self.runs):
            path.mkdir(parents=True, exist_ok=True)
        self.store = ExperimentStore(database)
        self.jobs = JobManager()
        self.env_path = env_path or root.parent.parent / ".env"
        self._backends: dict[tuple[str, str, str | None, str | None], LayaBackend] = {}
        self._backend_lock = threading.RLock()

    def _saved_env(self) -> dict[str, str]:
        values: dict[str, str] = {}
        if self.env_path.exists():
            for raw in self.env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() in self.ENV_KEYS:
                    value = value.strip()
                    try:
                        values[key.strip()] = json.loads(value) if value.startswith('"') else value
                    except json.JSONDecodeError:
                        values[key.strip()] = value
        return values

    def public_settings(self) -> dict[str, Any]:
        saved = self._saved_env()
        return {
            "base_url": os.getenv(
                "LAYA_LAB_LLM_BASE_URL",
                saved.get("LAYA_LAB_LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            ),
            "model": os.getenv(
                "LAYA_LAB_SPECIALIZATION_MODEL", saved.get("LAYA_LAB_SPECIALIZATION_MODEL", "")
            ),
            "has_api_key": bool(
                os.getenv("LAYA_LAB_LLM_API_KEY", saved.get("LAYA_LAB_LLM_API_KEY", ""))
            ),
        }

    def model_catalog(self) -> list[dict[str, Any]]:
        return model_catalog()

    def save_connection(self, config: ProviderConfig) -> dict[str, Any]:
        saved = self._saved_env()
        values = {
            "LAYA_LAB_LLM_BASE_URL": config.base_url,
            "LAYA_LAB_LLM_API_KEY": config.api_key or saved.get("LAYA_LAB_LLM_API_KEY", ""),
            "LAYA_LAB_SPECIALIZATION_MODEL": config.model,
        }
        existing = (
            self.env_path.read_text(encoding="utf-8").splitlines() if self.env_path.exists() else []
        )
        kept = [
            line
            for line in existing
            if not any(line.lstrip().startswith(f"{key}=") for key in self.ENV_KEYS)
        ]
        block = [f"{key}={json.dumps(value)}" for key, value in values.items()]
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_path.write_text(
            "\n".join(kept + ([""] if kept else []) + block) + "\n", encoding="utf-8"
        )
        os.chmod(self.env_path, 0o600)
        return self.public_settings()

    def provider(self, config: ProviderConfig) -> OpenAICompatibleProvider:
        saved = self._saved_env()
        api_key = config.api_key
        if not api_key:
            api_key = os.getenv("LAYA_LAB_LLM_API_KEY", saved.get("LAYA_LAB_LLM_API_KEY", ""))
        if not api_key:
            raise ValueError("An API key is required. Enter one or save it to .env.")
        return OpenAICompatibleProvider(
            config.model,
            base_url=config.base_url,
            api_key=api_key,
            cache_dir=Path("artifacts/cache/llm"),
        )

    def plan(self, request: PlanRequest) -> Job:
        def work(update):
            update(15, "Connecting to the LLM", f"Connecting to {request.provider.model}…")
            provider = self.provider(request.provider)
            update(45, "Designing the policy", "The LLM is structuring classes and boundaries.")
            task, record = infer_task(provider, request.description, request.labels, request.seed)
            update(90, "Preparing review", "The policy is ready to review.")
            return {
                "task": task.model_dump(mode="json"),
                "generation": record.model_dump(mode="json"),
            }

        return self.jobs.start("plan", work)

    def generate(self, request: GenerateRequest) -> Job:
        def work(update):
            spec_provider = self.provider(request.specialization_provider)
            benchmark_provider = self.provider(request.benchmark_provider)
            update(10, "Generating data", "Creating independent specialization and test data.")
            with ThreadPoolExecutor(max_workers=2) as pool:
                spec_future = pool.submit(
                    generate_specialization,
                    spec_provider,
                    request.task,
                    request.specialization_examples,
                    request.seed + 1,
                )
                benchmark_future = pool.submit(
                    generate_benchmark_splits,
                    benchmark_provider,
                    request.task,
                    {
                        "validation": request.validation_examples,
                        "hidden": request.hidden_examples,
                        "paraphrase": request.paraphrase_examples,
                        "hard": request.hard_examples,
                    },
                    request.seed + 50_000,
                )
                specialization, spec_record = spec_future.result()
                update(48, "Independent benchmark", "Finalizing hidden and robustness cases.")
                tests, benchmark_record = benchmark_future.result()
            update(82, "Leakage check", "Checking duplicates and split separation.")
            draft_id = f"{request.task.name}-{uuid.uuid4().hex[:8]}"
            examples, removed = remove_generated_leakage(specialization + tests)
            manifest = SuiteManifest(
                name=draft_id,
                tasks=[request.task.name],
                seed=request.seed,
                specialization_generation=spec_record,
                benchmark_generation=benchmark_record,
                metadata={
                    "created_with": "laya-studio",
                    "removed_leakage_collisions": removed,
                },
            )
            path = self.drafts / draft_id
            save_suite(path, manifest, [request.task], examples)
            _, _, splits = load_suite(path)
            update(95, "Ready for review", "The examples can now be edited.")
            return {
                "draft_id": draft_id,
                "task": request.task.model_dump(mode="json"),
                "splits": {
                    name: [row.model_dump(mode="json", exclude_none=True) for row in rows]
                    for name, rows in splits.items()
                },
            }

        return self.jobs.start("generate", work)

    def run(self, draft_id: str, request: RunRequest) -> Job:
        def work(update):
            update(5, "Manual review", "Applying your edits and checking for leakage.")
            examples = []
            for raw in request.examples:
                cleaned = dict(raw)
                cleaned.update(
                    task_name=request.task.name,
                    domain=request.task.domain,
                    content_hash=None,
                )
                examples.append(Example.model_validate(cleaned))
            draft_path = self.drafts / draft_id
            old_manifest, _, _ = load_suite(draft_path)
            manifest = old_manifest.model_copy(update={"tasks": [request.task.name]})
            save_suite(draft_path, manifest, [request.task], examples)
            update(
                18,
                "Loading Laya",
                f"Loading {request.model} on {request.device or 'auto'}.",
            )
            backend = self._backend(
                request.backend, request.model, request.device, request.revision
            )
            update(
                32,
                "Specializations",
                f"Comparing {len(request.methods)} training-free methods.",
            )
            run_id = benchmark_suite(
                draft_path,
                request.methods,
                backend,
                self.store,
                self.runs,
                request.seed,
                batch_size=64,
            )
            display_name = (request.name or request.task.name.replace("_", " ")).strip()
            self.store.set_run_name(run_id, display_name)
            update(82, "Selecting the best result", "Ranking by validation, then latency.")
            data = self.store.run(run_id)
            ranked = sorted(
                data["results"],
                key=lambda row: (
                    -row["metrics"]["splits"].get("validation", {}).get("accuracy", -1),
                    row["timings"].get("latency_per_sample_ms") or float("inf"),
                ),
            )
            if not ranked:
                raise RuntimeError("no strategy produced an exportable result")
            best = ranked[0]
            self.store.set_result_default(run_id, request.task.name, best["strategy"])
            specialization_id = f"{run_id}:{request.task.name}:{best['strategy']}"
            stored = self.store.specialization(specialization_id)
            task, fitted = load_saved_fitted(Path(stored["artifact_path"]), backend)
            export_path = self.exports / f"{task.name}-{best['strategy']}-{run_id[-8:]}"
            _, _, splits = load_suite(draft_path)
            probes = [row.input for row in splits["hidden"][:10]]
            update(90, "Fidelity export", "Reloading the artifact in a fresh process.")
            export_specialization(
                export_path,
                name=export_path.name,
                task=task,
                fitted=fitted,
                backend=backend,
                probes=probes,
            )
            baseline = next((row for row in ranked if row["strategy"] == "baseline"), None)
            baseline_score = (
                baseline["metrics"]["splits"].get("hidden", {}).get("accuracy")
                if baseline
                else None
            )
            public_ranking = []
            for row in ranked:
                score = row["metrics"]["splits"].get("hidden", {}).get("accuracy")
                public_ranking.append(
                    {
                        "strategy": row["strategy"],
                        "decision_component": row["decision_component"],
                        "accuracy": score,
                        "validation_accuracy": row["metrics"]["splits"]
                        .get("validation", {})
                        .get("accuracy"),
                        "delta": score - baseline_score
                        if score is not None and baseline_score is not None
                        else None,
                        "macro_f1": row["metrics"]["splits"].get("hidden", {}).get("macro_f1"),
                        "ece": row["metrics"]["splits"].get("hidden", {}).get("ece"),
                        "latency_ms": row["timings"].get("latency_per_sample_ms"),
                        "robustness": row["metrics"].get("robustness", {}),
                    }
                )
            update(98, "Finalizing", "The best artifact is ready to download.")
            return {
                "run_id": run_id,
                "best": public_ranking[0],
                "ranking": public_ranking,
                "export_path": str(export_path.resolve()),
                "specialization_id": specialization_id,
                "display_name": display_name,
            }

        return self.jobs.start("run", work)

    def library_run(self, run_id: str) -> dict[str, Any]:
        data = self.store.run(run_id)
        _manifest, tasks, splits = load_suite(Path(data["run"]["suite_path"]))
        examples = [
            row.model_dump(mode="json", exclude_none=True)
            for split in ("hidden", "paraphrase", "hard", "validation")
            for row in splits[split]
        ]
        return {
            **data,
            "tasks": [task.model_dump(mode="json") for task in tasks],
            "examples": examples,
        }

    def predict_result(self, run_id: str, request: PlaygroundRequest) -> dict[str, Any]:
        data = self.store.run(run_id)
        specialization_id = f"{run_id}:{request.task}:{request.strategy}"
        stored = self.store.specialization(specialization_id)
        device = "mps" if data["run"]["backend"] == "pytorch" else None
        backend = self._backend(
            data["run"]["backend"],
            data["run"]["base_model"],
            device,
            data["run"]["base_revision"],
        )
        _task, fitted = load_saved_fitted(Path(stored["artifact_path"]), backend)
        return fitted.predict([request.text])[0]

    def export_result(self, run_id: str, request: ExportResultRequest) -> str:
        data = self.store.run(run_id)
        stored = self.store.specialization(f"{run_id}:{request.task}:{request.strategy}")
        device = "mps" if data["run"]["backend"] == "pytorch" else None
        backend = self._backend(
            data["run"]["backend"],
            data["run"]["base_model"],
            device,
            data["run"]["base_revision"],
        )
        task, fitted = load_saved_fitted(Path(stored["artifact_path"]), backend)
        label = request.name or data["run"].get("display_name") or task.name
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-").lower() or task.name
        export_path = self.exports / f"{slug}-{request.strategy}-{run_id[-8:]}"
        if not export_path.exists():
            _manifest, _tasks, splits = load_suite(Path(data["run"]["suite_path"]))
            probes = [row.input for row in splits["hidden"] if row.task_name == request.task][:10]
            export_specialization(
                export_path,
                name=label,
                task=task,
                fitted=fitted,
                backend=backend,
                probes=probes,
            )
        return export_path.name

    def draft(self, draft_id: str) -> dict[str, Any]:
        _, tasks, splits = load_suite(self.drafts / draft_id)
        return {
            "draft_id": draft_id,
            "task": tasks[0].model_dump(mode="json"),
            "splits": {
                name: [row.model_dump(mode="json", exclude_none=True) for row in rows]
                for name, rows in splits.items()
            },
        }

    def _backend(
        self, name: str, model: str, device: str | None, revision: str | None = None
    ) -> LayaBackend:
        key = (name, model, device, revision)
        with self._backend_lock:
            if key not in self._backends:
                self._backends[key] = create_backend(name, model, device, revision)
            return self._backends[key]
