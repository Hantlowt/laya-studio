from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from .backends import LayaBackend, create_backend
from .io import write_json
from .schemas import DecisionType, ExportManifest, StrategyArtifact, TaskSpec, canonical_json
from .strategies import FittedStrategy, normalize, probability_rows, softmax

FORMAT_VERSION = 1


def export_specialization(
    output: Path,
    *,
    name: str,
    task: TaskSpec,
    fitted: FittedStrategy,
    backend: LayaBackend,
    probes: list[str],
    self_contained: bool = False,
    license_name: str | None = "Apache-2.0",
    attribution: str | None = "Laya model weights by Convai Innovations and upstream contributors.",
    verify: bool = True,
) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"export directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_model_path = None
    if self_contained:
        source = Path(backend.info.model_id)
        if not source.is_dir():
            raise ValueError(
                "--self-contained requires the base model to be an existing local directory"
            )
        destination = output / "base_model"
        shutil.copytree(source, destination)
        base_model_path = "base_model"
    manifest = ExportManifest(
        name=name,
        base_model=backend.info.model_id,
        base_revision=backend.info.revision,
        backend_compatibility=[backend.info.name],
        strategy=fitted.artifact.strategy,
        decision_schema=task.decision.model_dump(mode="json"),
        specialization_config=fitted.artifact.params,
        created_at=datetime.now(timezone.utc).isoformat(),
        package_version="0.1.0",
        self_contained=self_contained,
        base_model_path=base_model_path,
        license=license_name,
        attribution=attribution,
    )
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "specialization.json",
        {"task": task.model_dump(mode="json"), "artifact": fitted.artifact.model_dump(mode="json")},
    )
    arrays = {key: np.ascontiguousarray(value) for key, value in sorted(fitted.arrays.items())}
    save_tensors_deterministic(
        output / "vectors.safetensors",
        arrays,
        metadata={"format": "laya-specialization", "format_version": "1"},
    )
    write_json(
        output / "processor.json",
        {"normalization": "l2", "input": "utf-8 text", "schema_version": 1},
    )
    before = fitted.predict(probes)
    write_json(
        output / "fidelity.json", {"probes": probes, "expected": before, "rtol": 1e-5, "atol": 1e-6}
    )
    (output / "README.md").write_text(_artifact_readme(manifest, task))
    if verify:
        _fresh_process_fidelity(output, backend.info.name)
    return output


class SpecializedLaya:
    """Portable wrapper preserving Laya-like `predict` and `system_one` entry points."""

    def __init__(self, root: Path, backend: LayaBackend):
        self.root = root
        self.manifest = ExportManifest.model_validate_json((root / "manifest.json").read_text())
        if self.manifest.format_version != FORMAT_VERSION:
            raise ValueError(f"unsupported artifact version {self.manifest.format_version}")
        raw = json.loads((root / "specialization.json").read_text())
        self.task = TaskSpec.model_validate(raw["task"])
        self.artifact = StrategyArtifact.model_validate(raw["artifact"])
        self.arrays = load_file(str(root / "vectors.safetensors"))
        missing = set(self.artifact.arrays) - set(self.arrays)
        if missing:
            raise ValueError(f"artifact is missing tensors: {sorted(missing)}")
        self.backend = backend
        _assert_checkpoint_compatible(self.manifest, backend)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        backend: str | LayaBackend | None = None,
        device: str | None = None,
    ) -> "SpecializedLaya":
        root = Path(path)
        manifest = ExportManifest.model_validate_json((root / "manifest.json").read_text())
        if isinstance(backend, LayaBackend):
            runtime = backend
        else:
            backend_name = backend or manifest.backend_compatibility[0]
            model = (
                str(root / manifest.base_model_path)
                if manifest.self_contained and manifest.base_model_path
                else manifest.base_model
            )
            if str(backend_name) == "fake":
                tensors = load_file(str(root / "vectors.safetensors"))
                features = [
                    value
                    for key, value in tensors.items()
                    if value.ndim and key not in {"owners", "example_labels"}
                ]
                dimension = max((int(value.shape[-1]) for value in features), default=32)
                from .backends import FakeBackend

                runtime = FakeBackend(dimension=dimension, model_id=model)
            else:
                runtime = create_backend(
                    str(backend_name), model, device, revision=manifest.base_revision
                )
        return cls(root, runtime)

    def predict(self, state: str | dict | list, questions: dict | None = None) -> dict[str, Any]:
        text = state if isinstance(state, str) else canonical_json(state)
        row = self.predict_batch([text])[0]
        probabilities = row["probabilities"]
        decision_type = self.task.decision.type
        answer: dict[str, Any] = {
            "type": decision_type.value,
            "confidence": max(probabilities.values()),
        }
        if decision_type in (DecisionType.choice, DecisionType.ranking):
            answer.update(choice=row["label"], probabilities=probabilities)
        elif decision_type == DecisionType.noul:
            answer.update(noul=probabilities[self.task.decision.labels[1]])
        else:
            answer.update(
                score=sum(
                    i * probabilities[label] for i, label in enumerate(self.task.decision.labels)
                ),
                legend={str(i): label for i, label in enumerate(self.task.decision.labels)},
                probabilities={
                    str(i): probabilities[label]
                    for i, label in enumerate(self.task.decision.labels)
                },
            )
        return {
            "model": self.manifest.name,
            "answers": {"decision": answer},
            "usage": {"input_tokens": None, "output_tokens": 0},
        }

    system_one = predict

    def predict_batch(self, states: list[str]) -> list[dict[str, Any]]:
        return predict_saved(self.artifact, self.arrays, self.task, self.backend, states)


def predict_saved(
    artifact: StrategyArtifact,
    arrays: dict[str, np.ndarray],
    task: TaskSpec,
    backend: LayaBackend,
    states: list[str],
) -> list[dict[str, Any]]:
    name, params, labels, a = artifact.strategy, artifact.params, task.decision.labels, arrays
    if name == "baseline":
        return backend.predict_batch(states, task)
    if name == "prompt_only":
        updated = task.model_copy(deep=True)
        updated.decision.question = params["rewritten_instructions"]
        return backend.predict_batch(states, updated)
    if name in {"nearest_prototype", "multiclass_centroids"}:
        x = normalize(backend.embed(states))
        if params.get("mode", "mean") in {"mean", "medoid"} or name == "multiclass_centroids":
            logits = x @ a["prototypes"].T
        else:
            sims, owners = x @ a["example_embeddings"].T, a["example_labels"].astype(int)
            logits = np.full((len(x), len(labels)), -1.0, dtype=np.float32)
            if params["mode"] == "nearest":
                nearest = sims.argmax(1)
                for row, idx in enumerate(nearest):
                    logits[row, owners[idx]] = sims[row, idx]
            else:
                k = min(int(params["top_k"]), sims.shape[1])
                for row, indices in enumerate(np.argpartition(-sims, k - 1, axis=1)[:, :k]):
                    for idx in indices:
                        logits[row, owners[idx]] += max(float(sims[row, idx]), 0) + 1e-12
        return probability_rows(softmax(logits, params.get("temperature", 0.2)), labels)
    if name == "contrastive_vector":
        scores = normalize(backend.embed(states)) @ a["direction"] * params["strength"]
        positive = 1 / (1 + np.exp(-scores))
        return probability_rows(np.column_stack([1 - positive, positive]), labels)
    if name == "whitened_prototypes":
        x = normalize((backend.embed(states) - a["mean"]) @ a["whitening"])
        return probability_rows(softmax(x @ a["prototypes"].T, params["temperature"]), labels)
    if name == "residual_embedding_transform":
        x = normalize(normalize(backend.embed(states)) @ a["transform"])
        return probability_rows(softmax(x @ a["prototypes"].T, params["temperature"]), labels)
    if name == "multi_vector_steering":
        sims, owners = normalize(backend.embed(states)) @ a["vectors"].T, a["owners"].astype(int)
        logits = np.stack([sims[:, owners == i].max(1) for i in range(len(labels))], axis=1)
        return probability_rows(softmax(logits, params["temperature"]), labels)
    if name == "activation_steering":
        return backend.predict_steered_batch(
            states, task, a["task_vector"], params["strength"], params["layer"]
        )
    if name == "pairwise_ranking":
        labels = task.decision.labels
        totals = np.zeros((len(states), len(labels)), dtype=np.float32)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pair_task = task.model_copy(deep=True)
                pair_task.decision.type = DecisionType.choice
                pair_task.decision.labels = [labels[i], labels[j]]
                for row, value in enumerate(backend.predict_batch(states, pair_task)):
                    totals[row, labels.index(value["label"])] += 1.0
        probs = totals / np.maximum(totals.sum(1, keepdims=True), 1.0)
        return probability_rows(probs, labels)
    raise NotImplementedError(f"portable loading is not implemented for {name}")


def load_saved_fitted(path: Path, backend: LayaBackend) -> tuple[TaskSpec, FittedStrategy]:
    raw = json.loads((path / "specialization.json").read_text())
    checkpoint = raw.get("checkpoint")
    if checkpoint:
        expected = (
            checkpoint.get("backend"),
            checkpoint.get("model_id"),
            checkpoint.get("revision"),
        )
        actual = (backend.info.name, backend.info.model_id, backend.info.revision)
        if expected != actual:
            raise ValueError(
                "specialization checkpoint mismatch: "
                f"expected {expected[0]}:{expected[1]}@{expected[2]}, "
                f"got {actual[0]}:{actual[1]}@{actual[2]}"
            )
    task = TaskSpec.model_validate(raw["task"])
    artifact = StrategyArtifact.model_validate(raw["artifact"])
    arrays = load_file(str(path / "vectors.safetensors"))
    fitted = FittedStrategy(artifact=artifact, arrays=arrays)
    fitted._predict = lambda states: predict_saved(artifact, arrays, task, backend, states)
    return task, fitted


def _assert_checkpoint_compatible(manifest: ExportManifest, backend: LayaBackend) -> None:
    if backend.info.name not in manifest.backend_compatibility:
        raise ValueError(
            f"artifact backend mismatch: expected {manifest.backend_compatibility}, "
            f"got {backend.info.name}"
        )
    model_matches = backend.info.model_id == manifest.base_model
    if manifest.self_contained and manifest.base_model_path:
        model_matches = Path(backend.info.model_id).name == Path(manifest.base_model_path).name
    revision_matches = backend.info.revision == manifest.base_revision
    if not model_matches or not revision_matches:
        raise ValueError(
            "artifact checkpoint mismatch: "
            f"expected {manifest.base_model}@{manifest.base_revision}, "
            f"got {backend.info.model_id}@{backend.info.revision}"
        )


def load(path: str | Path, **kwargs: Any) -> SpecializedLaya:
    return SpecializedLaya.from_pretrained(path, **kwargs)


def save_tensors_deterministic(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, str] | None = None
) -> None:
    """Write the documented SafeTensors layout with canonical header and tensor ordering."""
    dtype_codes = {
        np.dtype("float16"): "F16",
        np.dtype("float32"): "F32",
        np.dtype("float64"): "F64",
        np.dtype("int8"): "I8",
        np.dtype("int16"): "I16",
        np.dtype("int32"): "I32",
        np.dtype("int64"): "I64",
        np.dtype("uint8"): "U8",
        np.dtype("bool"): "BOOL",
    }
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = dict(sorted(metadata.items()))
    chunks: list[bytes] = []
    offset = 0
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        native = array.dtype.newbyteorder("=")
        if native not in dtype_codes:
            raise TypeError(f"unsupported SafeTensors dtype for {name}: {array.dtype}")
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        data = array.tobytes(order="C")
        header[name] = {
            "dtype": dtype_codes[native],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def verify_fidelity(path: Path, backend: str | LayaBackend | None = None) -> None:
    agent = SpecializedLaya.from_pretrained(path, backend=backend)
    fidelity = json.loads((path / "fidelity.json").read_text())
    actual = agent.predict_batch(fidelity["probes"])
    _assert_prediction_close(fidelity["expected"], actual, fidelity["rtol"], fidelity["atol"])


def _assert_prediction_close(
    expected: list[dict], actual: list[dict], rtol: float, atol: float
) -> None:
    if [x["label"] for x in expected] != [x["label"] for x in actual]:
        raise AssertionError("export fidelity failed: labels changed after reload")
    for before, after in zip(expected, actual):
        keys = sorted(before["probabilities"])
        if keys != sorted(after["probabilities"]):
            raise AssertionError("export fidelity failed: probability labels changed")
        if not np.allclose(
            [before["probabilities"][k] for k in keys],
            [after["probabilities"][k] for k in keys],
            rtol=rtol,
            atol=atol,
        ):
            raise AssertionError("export fidelity failed: probabilities changed after reload")


def _fresh_process_fidelity(path: Path, backend_name: str) -> None:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "laya_studio.fidelity", str(path), "--backend", backend_name]
    result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=300)
    if result.returncode:
        raise AssertionError(
            f"fresh-process export fidelity failed:\n{result.stderr or result.stdout}"
        )


def _artifact_readme(manifest: ExportManifest, task: TaskSpec) -> str:
    return f"""# {manifest.name}

Portable Laya specialization artifact (format version {manifest.format_version}).

- Base model: `{manifest.base_model}`
- Strategy: `{manifest.strategy}`
- Task: {task.description}
- Decision labels: {", ".join(task.decision.labels)}
- License: {manifest.license or "See base model metadata"}

```python
from laya_studio import SpecializedLaya
agent = SpecializedLaya.from_pretrained(".")
result = agent.predict("your input")
```

{manifest.attribution or ""}
"""
