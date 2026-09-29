from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .models import resolve_revision
from .schemas import DecisionType, TaskSpec


@dataclass
class BackendInfo:
    name: str
    model_id: str
    revision: str | None = None
    device: str | None = None
    supports_activation_steering: bool = False


class LayaBackend(ABC):
    info: BackendInfo

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, states: Sequence[str], task: TaskSpec) -> list[dict[str, Any]]:
        raise NotImplementedError

    def predict_steered_batch(
        self, states: Sequence[str], task: TaskSpec, vector: np.ndarray, alpha: float, layer: int
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(f"activation steering is unsupported by {self.info.name}")

    def freeze(self) -> None:
        """Freeze model parameters if the runtime exposes them."""

    def _cached_embed(
        self, texts: Sequence[str], compute: Any, cache: dict[str, np.ndarray]
    ) -> np.ndarray:
        rows = [str(text) for text in texts]
        missing = list(dict.fromkeys(text for text in rows if text not in cache))
        if missing:
            values = np.asarray(compute(missing), dtype=np.float32)
            for text, value in zip(missing, values):
                cached = np.array(value, dtype=np.float32, copy=True)
                cached.setflags(write=False)
                cache[text] = cached
        if not rows:
            dimension = next((len(value) for value in cache.values()), 0)
            return np.empty((0, dimension), dtype=np.float32)
        return np.stack([cache[text] for text in rows])


def task_question(
    task: TaskSpec, label_order: list[str] | None = None, instructions: str | None = None
) -> dict:
    labels = label_order or task.decision.labels
    descriptions = task.decision.class_descriptions
    qtype = task.decision.type
    if qtype in (DecisionType.choice, DecisionType.ranking):
        criteria = {label: descriptions.get(label) for label in labels}
        typ = "choice"
    elif qtype == DecisionType.score:
        criteria = [descriptions.get(label, label) for label in labels]
        typ = "score"
    else:
        typ = "noul"
        criteria = {
            "false": descriptions.get(labels[0], labels[0]),
            "true": descriptions.get(labels[1], labels[1]),
        }
    return {
        "decision": {
            "type": typ,
            "instructions": instructions or task.decision.question or task.description,
            "criteria": criteria,
            **({"labels": {"false": labels[0], "true": labels[1]}} if typ == "noul" else {}),
        }
    }


def parse_laya_result(result: dict, task: TaskSpec) -> dict[str, Any]:
    answer = result["answers"]["decision"]
    labels = task.decision.labels
    if task.decision.type in (DecisionType.choice, DecisionType.ranking):
        probs = {label: float(answer["probabilities"].get(label, 0.0)) for label in labels}
        label = answer["choice"]
    elif task.decision.type == DecisionType.noul:
        p = float(answer["noul"])
        probs = {labels[0]: 1.0 - p, labels[1]: p}
        label = labels[int(p >= 0.5)]
    else:
        raw = answer.get("probabilities", {})
        probs = {label: float(raw.get(str(i), 0.0)) for i, label in enumerate(labels)}
        label = labels[int(np.argmax(list(probs.values())))]
    return {"label": label, "probabilities": probs, "raw": result}


class PyTorchLayaBackend(LayaBackend):
    def __init__(
        self,
        model_id: str = "convaiinnovations/laya",
        device: str | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ):
        import laya

        resolved_revision = resolve_revision(model_id, revision)
        started = time.perf_counter()
        self.agent = laya.load(model_id, device=device, revision=resolved_revision, **kwargs)
        self.cold_load_seconds = time.perf_counter() - started
        self._embed = laya.embed_fn_from_agent(self.agent)
        self._lock = threading.RLock()
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._layers = self._find_layers()
        self.info = BackendInfo(
            "pytorch",
            model_id,
            revision=resolved_revision,
            device=str(self.agent.device),
            supports_activation_steering=bool(self._layers),
        )
        self.freeze()

    def freeze(self) -> None:
        self.agent.model.eval()
        for parameter in self.agent.model.parameters():
            parameter.requires_grad_(False)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            return self._cached_embed(texts, self._embed, self._embedding_cache)

    def predict_batch(self, states: Sequence[str], task: TaskSpec) -> list[dict[str, Any]]:
        results = self.agent.predict_batch(list(states), task_question(task))
        return [parse_laya_result(x, task) for x in results]

    def _find_layers(self) -> list[Any]:
        enc = self.agent.model.encoder
        for path in (("layers",), ("encoder", "layer"), ("model", "layers")):
            obj = enc
            for name in path:
                obj = getattr(obj, name, None)
                if obj is None:
                    break
            if obj is not None:
                try:
                    return list(obj)
                except TypeError:
                    pass
        return []

    def predict_steered_batch(
        self, states: Sequence[str], task: TaskSpec, vector: np.ndarray, alpha: float, layer: int
    ) -> list[dict[str, Any]]:
        if not self._layers:
            return super().predict_steered_batch(states, task, vector, alpha, layer)
        import torch

        selected = self._layers[layer]
        v = torch.as_tensor(vector, dtype=self.agent.dtype, device=self.agent.device)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            def steer(tensor: Any) -> Any:
                if torch.is_tensor(tensor) and tensor.ndim == 3 and tensor.shape[-1] == v.numel():
                    return tensor + alpha * v[None, None, :]
                return tensor

            if torch.is_tensor(output):
                return steer(output)
            if isinstance(output, tuple):
                return (steer(output[0]), *output[1:])
            return output

        with self._lock:
            handle = selected.register_forward_hook(hook)
            try:
                return self.predict_batch(states, task)
            finally:
                handle.remove()


class MLXLayaBackend(LayaBackend):
    def __init__(
        self,
        model_id: str = "aac6fef/laya-mlx",
        device: str | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ):
        import laya_mlx

        resolved_revision = resolve_revision(model_id, revision)
        started = time.perf_counter()
        self.agent = laya_mlx.load(model_id, device=device, revision=resolved_revision, **kwargs)
        self.cold_load_seconds = time.perf_counter() - started
        self._embed = laya_mlx.embed_fn_from_agent(self.agent)
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self.info = BackendInfo(
            "mlx",
            model_id,
            revision=resolved_revision,
            device=str(device),
            supports_activation_steering=False,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            return self._cached_embed(texts, self._embed, self._embedding_cache)

    def predict_batch(self, states: Sequence[str], task: TaskSpec) -> list[dict[str, Any]]:
        return [
            parse_laya_result(self.agent.predict(state, task_question(task)), task)
            for state in states
        ]


class FakeBackend(LayaBackend):
    """Deterministic backend for unit tests and plumbing smoke runs, never scientific evidence."""

    def __init__(self, dimension: int = 32, model_id: str = "fake-hash-v1") -> None:
        self.dimension = dimension
        self.info = BackendInfo("fake", model_id, revision="1", device="cpu")
        self.cold_load_seconds = 0.0
        self._embedding_cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached_embed(texts, self._embed_uncached, self._embedding_cache)

    def _embed_uncached(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = str(text).casefold().split()
            for token in tokens:
                digest = hashlib.sha256(token.encode()).digest()
                for i in range(0, min(len(digest), self.dimension)):
                    vec[i] += (digest[i] / 127.5) - 1.0
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm else vec)
        return np.stack(rows) if rows else np.empty((0, self.dimension), dtype=np.float32)

    def predict_batch(self, states: Sequence[str], task: TaskSpec) -> list[dict[str, Any]]:
        label_vectors = self.embed(
            [
                f"{label} {task.decision.class_descriptions.get(label, '')}"
                for label in task.decision.labels
            ]
        )
        x = self.embed(states)
        logits = x @ label_vectors.T
        probs = _softmax(logits)
        return [
            {
                "label": task.decision.labels[int(np.argmax(row))],
                "probabilities": dict(zip(task.decision.labels, map(float, row))),
                "raw": {"model": "fake-hash-v1"},
            }
            for row in probs
        ]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def create_backend(
    name: str,
    model_id: str | None = None,
    device: str | None = None,
    revision: str | None = None,
) -> LayaBackend:
    if name == "pytorch":
        return PyTorchLayaBackend(
            model_id or "convaiinnovations/laya", device=device, revision=revision
        )
    if name == "mlx":
        return MLXLayaBackend(model_id or "aac6fef/laya-mlx", device=device, revision=revision)
    if name == "fake":
        return FakeBackend(model_id=model_id or "fake-hash-v1")
    raise ValueError(f"unknown backend {name!r}")
