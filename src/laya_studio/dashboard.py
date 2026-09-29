from __future__ import annotations

import shutil
from pathlib import Path

from .store import ExperimentStore
from .studio import (
    DefaultRequest,
    ExportResultRequest,
    GenerateRequest,
    KeepRequest,
    NameRequest,
    PlanRequest,
    PlaygroundRequest,
    RunRequest,
    SettingsRequest,
    StudioService,
)


def create_app(database: Path):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, Response
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("install laya-studio[dashboard] to use the dashboard") from exc

    app = FastAPI(title="Laya Studio", docs_url="/api/docs", redoc_url=None)
    store = ExperimentStore(database)
    project_root = Path(__file__).resolve().parents[2]
    static_root = Path(__file__).resolve().parent / "static"
    studio = StudioService(project_root / "experiments" / "studio", database, project_root / ".env")
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/create", response_class=HTMLResponse)
    @app.get("/runs", response_class=HTMLResponse)
    @app.get("/library", response_class=HTMLResponse)
    @app.get("/playground", response_class=HTMLResponse)
    def shell():
        return _shell()

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.post("/api/studio/plan")
    def start_plan(request: PlanRequest):
        return studio.plan(request).public()

    @app.get("/api/studio/settings")
    def get_settings():
        return studio.public_settings()

    @app.get("/api/studio/models")
    def get_models():
        return studio.model_catalog()

    @app.post("/api/studio/settings")
    def save_settings(request: SettingsRequest):
        return studio.save_connection(request.provider)

    @app.post("/api/studio/generate")
    def start_generation(request: GenerateRequest):
        return studio.generate(request).public()

    @app.post("/api/studio/drafts/{draft_id}/run")
    def start_run(draft_id: str, request: RunRequest):
        if Path(draft_id).name != draft_id:
            raise HTTPException(400, "Invalid draft ID")
        if not (studio.drafts / draft_id).is_dir():
            raise HTTPException(404, "Draft not found")
        return studio.run(draft_id, request).public()

    @app.get("/api/studio/drafts/{draft_id}")
    def get_draft(draft_id: str):
        if Path(draft_id).name != draft_id:
            raise HTTPException(400, "Invalid draft ID")
        try:
            return studio.draft(draft_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, "Draft not found") from exc

    @app.get("/api/studio/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return studio.jobs.get(job_id).public()
        except KeyError as exc:
            raise HTTPException(404, "Unknown job") from exc

    @app.get("/api/studio/exports/{name}")
    def download_export(name: str):
        if Path(name).name != name:
            raise HTTPException(400, "Invalid export name")
        source = studio.exports / name
        if not source.is_dir():
            raise HTTPException(404, "Export not found")
        archive_base = studio.exports / f".{name}-download"
        archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=source))
        return FileResponse(archive, filename=f"{name}.zip", media_type="application/zip")

    @app.get("/api/runs")
    def api_runs():
        return store.list_runs()

    @app.get("/api/library/runs/{run_id}")
    def library_run(run_id: str):
        try:
            return studio.library_run(run_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/library/runs/{run_id}/name")
    def rename_run(run_id: str, request: NameRequest):
        store.set_run_name(run_id, request.name)
        return {"name": request.name}

    @app.post("/api/library/runs/{run_id}/keep")
    def keep_result(run_id: str, request: KeepRequest):
        try:
            store.set_result_kept(run_id, request.task, request.strategy, request.kept)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"kept": request.kept}

    @app.post("/api/library/runs/{run_id}/default")
    def default_result(run_id: str, request: DefaultRequest):
        try:
            store.set_result_default(run_id, request.task, request.strategy)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"strategy": request.strategy}

    @app.post("/api/library/runs/{run_id}/predict")
    def playground_predict(run_id: str, request: PlaygroundRequest):
        try:
            return studio.predict_result(run_id, request)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/library/runs/{run_id}/export")
    def export_result(run_id: str, request: ExportResultRequest):
        try:
            return {"export_name": studio.export_result(run_id, request)}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str):
        try:
            return store.run(run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/runs/{run_id}/mistakes")
    def api_mistakes(run_id: str, task: str, strategy: str):
        return store.mistakes(run_id, task, strategy)

    return app


def _shell() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="theme-color" content="#0d1b1e">
  <title>Laya Studio</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <a class="brand" href="/" data-nav="home" aria-label="Laya Studio home">
        <span class="brand-mark">L</span><span><strong>Laya</strong><small>Studio</small></span>
      </a>
      <nav aria-label="Main navigation">
        <a href="/" data-nav="home"><span class="nav-icon">⌂</span>Overview</a>
        <a href="/create" data-nav="create"><span class="nav-icon">✦</span>New specialization</a>
        <a href="/library" data-nav="library"><span class="nav-icon">▦</span>Model library</a>
        <a href="/playground" data-nav="playground"><span class="nav-icon">▷</span>Playground</a>
        <a href="/runs" data-nav="runs"><span class="nav-icon">◫</span>Experiments</a>
      </nav>
      <div class="sidebar-note">
        <span class="status-dot"></span>
        <div><strong>Laya ready</strong><small>PyTorch · Apple Silicon</small></div>
      </div>
    </aside>
    <main id="app" tabindex="-1"><div class="page-loading"><span></span>Loading…</div></main>
  </div>
  <div id="toast-region" aria-live="polite"></div>
  <script src="/static/app.js" defer></script>
</body>
</html>"""
