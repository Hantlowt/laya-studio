import pytest

from laya_studio.artifact import SpecializedLaya, export_specialization, verify_fidelity
from laya_studio.backends import FakeBackend
from laya_studio.strategies import PrototypeStrategy


def test_export_reload_fidelity_and_deterministic_tensors(
    tmp_path, binary_task, rows, fake_backend
):
    spec = [x for x in rows if x.split == "specialization"]
    fitted = PrototypeStrategy().fit(binary_task, spec, [], fake_backend)
    first, second = tmp_path / "first", tmp_path / "second"
    probes = ["Production unavailable", "Profile help"]
    export_specialization(
        first, name="test", task=binary_task, fitted=fitted, backend=fake_backend, probes=probes
    )
    export_specialization(
        second, name="test", task=binary_task, fitted=fitted, backend=fake_backend, probes=probes
    )
    assert (first / "vectors.safetensors").read_bytes() == (
        second / "vectors.safetensors"
    ).read_bytes()
    verify_fidelity(first, "fake")
    loaded = SpecializedLaya.from_pretrained(first, backend="fake")
    assert (
        loaded.predict("Production unavailable")["answers"]["decision"]["choice"]
        in binary_task.decision.labels
    )


def test_artifact_version_rejected(tmp_path, binary_task, rows, fake_backend):
    spec = [x for x in rows if x.split == "specialization"]
    fitted = PrototypeStrategy().fit(binary_task, spec, [], fake_backend)
    root = tmp_path / "bad-version"
    export_specialization(
        root,
        name="test",
        task=binary_task,
        fitted=fitted,
        backend=fake_backend,
        probes=["x"],
        verify=False,
    )
    manifest = (
        (root / "manifest.json").read_text().replace('"format_version": 1', '"format_version": 2')
    )
    (root / "manifest.json").write_text(manifest)
    with pytest.raises(Exception, match="format_version"):
        SpecializedLaya.from_pretrained(root, backend="fake")


def test_artifact_rejects_a_different_checkpoint(tmp_path, binary_task, rows, fake_backend):
    spec = [x for x in rows if x.split == "specialization"]
    fitted = PrototypeStrategy().fit(binary_task, spec, [], fake_backend)
    root = tmp_path / "locked"
    export_specialization(
        root,
        name="locked",
        task=binary_task,
        fitted=fitted,
        backend=fake_backend,
        probes=["probe"],
        verify=False,
    )
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        SpecializedLaya.from_pretrained(root, backend=FakeBackend(model_id="other-model"))
