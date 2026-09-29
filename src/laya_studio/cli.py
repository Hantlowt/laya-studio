from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from .artifact import export_specialization, load_saved_fitted
from .backends import create_backend
from .dashboard import create_app
from .evaluation import aggregate_run, benchmark_suite
from .generation import (
    build_suite,
    generate_benchmark_splits,
    generate_specialization,
    infer_task,
    invent_task_catalog,
)
from .io import load_suite, read_yaml
from .models import ensure_checkpoint, model_catalog, resolve_revision
from .providers import OpenAICompatibleProvider
from .schemas import TaskSpec
from .store import ExperimentStore
from .strategies import create_strategy

app = typer.Typer(
    no_args_is_help=True, help="Leakage-resistant specialization experiments for frozen Laya."
)


@app.command("models")
def models_command(download: bool = typer.Option(False, help="Download every listed checkpoint")):
    """List built-in vanilla checkpoints and their immutable Hub revisions."""
    rows = []
    for item in model_catalog():
        revision = resolve_revision(item["model_id"])
        row = {**item, "revision": revision}
        if download:
            row["local_path"] = ensure_checkpoint(item["model_id"], revision)
        rows.append(row)
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


def _store(path: Path) -> ExperimentStore:
    return ExperimentStore(path)


def _provider(model: str | None) -> OpenAICompatibleProvider:
    if not model:
        raise typer.BadParameter("set a provider model using the CLI option or LAYA_LAB_*_MODEL")
    return OpenAICompatibleProvider(model)


@app.command()
def generate(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("benchmarks/generated-suite"),
    domains: Annotated[int, typer.Option(min=1)] = 10,
    tasks_per_domain: Annotated[int, typer.Option(min=1)] = 5,
    specialization_examples: Annotated[int, typer.Option(min=2)] = 40,
    validation_examples: Annotated[int, typer.Option(min=2)] = 50,
    test_examples: Annotated[int, typer.Option(min=2)] = 300,
    paraphrase_examples: Annotated[int, typer.Option(min=2)] = 60,
    hard_examples: Annotated[int, typer.Option(min=2)] = 60,
    repetitions: Annotated[int, typer.Option(min=1)] = 1,
    specialization_model: Optional[str] = typer.Option(None),
    benchmark_model: Optional[str] = typer.Option(None),
    style_control_model: Optional[str] = typer.Option(
        None, help="Optional second benchmark style/model for leakage measurement"
    ),
    seed: int = 0,
):
    """Invent domains/tasks and generate independent specialization and test splits."""
    spec_model = specialization_model or os.getenv("LAYA_LAB_SPECIALIZATION_MODEL")
    bench_model = benchmark_model or os.getenv("LAYA_LAB_BENCHMARK_MODEL") or spec_model
    spec_provider, benchmark_provider = _provider(spec_model), _provider(bench_model)
    style_provider = _provider(style_control_model) if style_control_model else None
    for repetition in range(repetitions):
        repetition_seed = seed + repetition * 1_000_003
        destination = output if repetitions == 1 else output / f"seed-{repetition_seed}"
        catalog = invent_task_catalog(
            benchmark_provider, domains, tasks_per_domain, repetition_seed
        )
        build_suite(
            destination,
            destination.name,
            catalog,
            spec_provider,
            benchmark_provider,
            style_provider,
            specialization_examples,
            {
                "validation": validation_examples,
                "hidden": test_examples,
                "paraphrase": paraphrase_examples,
                "hard": hard_examples,
            },
            repetition_seed,
        )
        typer.echo(f"Wrote validated benchmark suite to {destination}")


@app.command()
def benchmark(
    suite: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    methods: str = "baseline,prompt_only,nearest_prototype,contrastive_vector,multiclass_centroids,whitened_prototypes,residual_embedding_transform,activation_steering,multi_vector_steering,pairwise_ranking",
    backend: str = os.getenv("LAYA_LAB_BACKEND", "pytorch"),
    model: str = os.getenv("LAYA_LAB_MODEL", "convaiinnovations/laya"),
    revision: Optional[str] = typer.Option(None, help="Hub commit/tag; pinned if omitted"),
    device: Optional[str] = os.getenv("LAYA_LAB_DEVICE") or None,
    database: Path = Path("experiments/laya_lab.sqlite3"),
    artifacts: Path = Path("experiments/runs"),
    seed: int = 0,
    seeds: Optional[str] = typer.Option(None, help="Comma-separated repeated-run seeds"),
    batch_size: int = 64,
):
    """Run all methods against the same immutable hidden examples."""
    runtime = create_backend(backend, model, device, revision)
    run_seeds = [int(x.strip()) for x in seeds.split(",")] if seeds else [seed]
    for run_seed in run_seeds:
        run_id = benchmark_suite(
            suite,
            [x.strip() for x in methods.split(",") if x.strip()],
            runtime,
            _store(database),
            artifacts,
            run_seed,
            batch_size,
        )
        typer.echo(run_id)


@app.command()
def report(
    run_id: str,
    database: Path = Path("experiments/laya_lab.sqlite3"),
    bootstrap_samples: int = 0,
    holdout_domain: Optional[str] = typer.Option(None),
    output: Optional[Path] = None,
):
    """Report paired task-level statistics against baseline."""
    store = _store(database)
    data = store.run(run_id)
    payload = {
        "run": data["run"],
        "summary": aggregate_run(
            data, bootstrap_samples=bootstrap_samples, holdout_domain=holdout_domain
        ),
        "results": data["results"],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        output.write_text(text + "\n")
        typer.echo(str(output))
    else:
        typer.echo(text)


@app.command()
def compare(run_a: str, run_b: str, database: Path = Path("experiments/laya_lab.sqlite3")):
    """Compare matching task/strategy accuracy between two completed runs."""
    store = _store(database)
    left, right = store.run(run_a), store.run(run_b)
    index = {(x["task_name"], x["strategy"]): x for x in left["results"]}
    rows = []
    for value in right["results"]:
        key = (value["task_name"], value["strategy"])
        if key in index:
            a = index[key]["metrics"]["overall"]["accuracy"]
            b = value["metrics"]["overall"]["accuracy"]
            rows.append(
                {
                    "task": key[0],
                    "strategy": key[1],
                    "run_a": a,
                    "run_b": b,
                    "delta_b_minus_a": b - a,
                }
            )
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


@app.command()
def specialize(
    task: Optional[str] = typer.Option(None, help="Natural-language task request"),
    task_file: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    labels: Optional[str] = typer.Option(None, help="Comma-separated labels for --task"),
    examples: int = 50,
    method: str = "multi_vector_steering",
    output: Path = Path("exports/specialized-laya"),
    backend: str = os.getenv("LAYA_LAB_BACKEND", "pytorch"),
    model: str = os.getenv("LAYA_LAB_MODEL", "convaiinnovations/laya"),
    revision: Optional[str] = typer.Option(None, help="Hub commit/tag; pinned if omitted"),
    device: Optional[str] = os.getenv("LAYA_LAB_DEVICE") or None,
    specialization_model: Optional[str] = None,
    benchmark_model: Optional[str] = None,
    seed: int = 0,
):
    """Generate synthetic examples, fit a strategy, and export a portable artifact."""
    spec_provider = _provider(specialization_model or os.getenv("LAYA_LAB_SPECIALIZATION_MODEL"))
    benchmark_provider = _provider(
        benchmark_model or os.getenv("LAYA_LAB_BENCHMARK_MODEL") or spec_provider.model
    )
    if task_file:
        task_spec = read_yaml(task_file, TaskSpec)
    elif task and labels:
        task_spec, _ = infer_task(
            spec_provider, task, [x.strip() for x in labels.split(",") if x.strip()], seed
        )
    else:
        raise typer.BadParameter("provide --task-file, or both --task and --labels")
    spec_rows, _ = generate_specialization(spec_provider, task_spec, examples, seed + 1)
    validation, _ = generate_benchmark_splits(
        benchmark_provider, task_spec, {"validation": max(20, examples // 2)}, seed + 50_000
    )
    runtime = create_backend(backend, model, device, revision)
    fitted = create_strategy(method).fit(task_spec, spec_rows, validation, runtime)
    probes = [x.input for x in validation[:10]] or [x.input for x in spec_rows[:10]]
    export_specialization(
        output, name=output.name, task=task_spec, fitted=fitted, backend=runtime, probes=probes
    )
    typer.echo(str(output))


@app.command("export")
def export_command(
    specialization_id: str,
    output: Annotated[Path, typer.Option("--output", "-o")],
    database: Path = Path("experiments/laya_lab.sqlite3"),
    self_contained: bool = False,
    device: Optional[str] = None,
):
    """Export a stored run specialization and enforce fresh-process fidelity."""
    store = _store(database)
    record = store.specialization(specialization_id)
    runtime = create_backend(
        record["backend"], record["base_model"], device, record["base_revision"]
    )
    task, fitted = load_saved_fitted(Path(record["artifact_path"]), runtime)
    _manifest, _tasks, splits = load_suite(Path(record["suite_path"]))
    probes = [x.input for x in splits["hidden"] if x.task_name == task.name][:10]
    export_specialization(
        output,
        name=output.name,
        task=task,
        fitted=fitted,
        backend=runtime,
        probes=probes,
        self_contained=self_contained,
    )
    typer.echo(str(output))


@app.command()
def serve(
    database: Path = Path("experiments/laya_lab.sqlite3"), host: str = "127.0.0.1", port: int = 8787
):
    """Serve the optional read-only local dashboard."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("install laya-studio[dashboard]") from exc
    uvicorn.run(create_app(database), host=host, port=port)


if __name__ == "__main__":
    app()
