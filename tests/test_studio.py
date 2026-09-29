import json
import stat

import pytest
from fastapi.testclient import TestClient

from laya_studio.dashboard import create_app
from laya_studio.generation import infer_task
from laya_studio.providers import StaticProvider
from laya_studio.store import ExperimentStore, now_iso
from laya_studio.studio import ProviderConfig, StudioService


def test_studio_shell_and_assets(tmp_path):
    client = TestClient(create_app(tmp_path / "experiments.sqlite3"))
    shell = client.get("/")
    assert shell.status_code == 200
    assert "Laya Studio" in shell.text
    assert "<pre>" not in shell.text
    assert client.get("/create").status_code == 200
    assert client.get("/runs").status_code == 200
    assert client.get("/library").status_code == 200
    assert client.get("/playground").status_code == 200
    assert "New specialization" in client.get("/static/app.js").text
    assert "Model library" in client.get("/static/app.js").text
    assert "--teal" in client.get("/static/app.css").text
    assert client.get("/api/runs").json() == []
    models = client.get("/api/studio/models").json()
    assert {row["model_id"] for row in models} >= {
        "aac6fef/laya-mlx",
        "aac6fef/laya-multilingual-mlx",
        "aac6fef/laya-typed-decisions-mlx",
        "convaiinnovations/laya",
        "convaiinnovations/laya-multilingual",
        "convaiinnovations/laya-typed-decisions",
    }
    assert "Custom Hugging Face ID" in client.get("/static/app.js").text


def test_env_connection_is_private_and_reloadable(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("UNRELATED=value\n", encoding="utf-8")
    service = StudioService(tmp_path / "studio", tmp_path / "lab.sqlite3", env_path)

    public = service.save_connection(
        ProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4.1-mini",
            api_key="test-secret",
        )
    )

    assert public == {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4.1-mini",
        "has_api_key": True,
    }
    assert "test-secret" not in str(public)
    assert "UNRELATED=value" in env_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    provider = service.provider(
        ProviderConfig(base_url=public["base_url"], model=public["model"], api_key="")
    )
    assert provider.api_key == "test-secret"


def test_task_labels_can_be_inferred():
    response = {
        "task": {
            "name": "inferred_task",
            "description": "Classify a request using labels inferred by the provider.",
            "domain": "test",
            "decision": {
                "type": "choice",
                "labels": ["ALLOW", "BLOCK"],
                "class_descriptions": {"ALLOW": "safe", "BLOCK": "unsafe"},
            },
        }
    }
    task, record = infer_task(
        StaticProvider([response]), "Infer appropriate policy labels", None, 3
    )
    assert task.decision.labels == ["ALLOW", "BLOCK"]
    assert "Infer 2 to 6" in record.prompt


def test_library_names_and_kept_methods_are_persistent(tmp_path):
    store = ExperimentStore(tmp_path / "lab.sqlite3")
    store.create_run(
        {
            "id": "run-1",
            "created_at": now_iso(),
            "status": "completed",
            "suite_path": str(tmp_path / "suite"),
            "base_model": "fake/laya",
            "base_revision": "test",
            "backend": "fake",
            "seed": 7,
            "environment_json": json.dumps({}),
            "git_commit": None,
        }
    )
    store.add_result(
        {
            "run_id": "run-1",
            "task_name": "ticket_priority",
            "domain": "support",
            "strategy": "nearest_prototype",
            "decision_component": "embedding_classifier",
            "strategy_params": {},
            "metrics": {"splits": {"hidden": {"accuracy": 0.8}}},
            "timings": {},
        }
    )

    store.set_run_name("run-1", "Priority Assistant")
    store.set_result_default("run-1", "ticket_priority", "nearest_prototype")

    run = store.run("run-1")
    assert run["run"]["display_name"] == "Priority Assistant"
    assert run["results"][0]["kept"] is True
    assert run["results"][0]["is_default"] is True
    assert store.list_runs()[0]["kept_count"] == 1
    assert store.list_runs()[0]["task_count"] == 1
    assert store.list_runs()[0]["default_strategy"] == "nearest_prototype"
    with pytest.raises(ValueError, match="another default"):
        store.set_result_kept("run-1", "ticket_priority", "nearest_prototype", False)
