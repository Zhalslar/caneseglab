from __future__ import annotations

import asyncio
import json
import mimetypes
import uuid
import webbrowser
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from threading import Timer
from typing import Any, Callable

from .artifacts import latest_run_dir
from .config import ProjectConfig
from .core import SegmentationProject
from .datasets import build_dataset_paths, iter_dataset_dirs
from .services.log_service import logger

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web_static"
WORKSPACE_ROOT = BASE_DIR.parent.resolve()


@dataclass
class JobRecord:
    """后台任务记录。"""

    job_id: str
    kind: str
    status: str
    created_at: str
    payload: dict[str, Any]
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress_completed: int = 0
    progress_total: int = 0
    progress_percent: float = 0.0
    progress_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    """简单后台任务管理器。"""

    def __init__(self, single_job_mode: bool = True) -> None:
        self._single_job_mode = single_job_mode
        self._jobs: dict[str, JobRecord] = {}
        self._active_job_ids: set[str] = set()
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._lock = asyncio.Lock()

    async def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        runner: Callable[[Callable[[int, int, str | None], None]], dict[str, Any]],
    ) -> JobRecord:
        async with self._lock:
            if self._single_job_mode and self._active_job_ids:
                raise RuntimeError("当前已有任务在运行，请等待完成后再提交新任务。")

            job = JobRecord(
                job_id=uuid.uuid4().hex[:12],
                kind=kind,
                status="queued",
                created_at=_now_text(),
                payload=_serialize(payload),
            )
            self._jobs[job.job_id] = job
            self._active_job_ids.add(job.job_id)
            self._publish_locked()

        asyncio.create_task(self._run_job(job, runner))
        return job

    async def _run_job(
        self,
        job: JobRecord,
        runner: Callable[[Callable[[int, int, str | None], None]], dict[str, Any]],
    ) -> None:
        loop = asyncio.get_running_loop()
        job.status = "running"
        job.started_at = _now_text()
        self._publish()

        try:
            result = await asyncio.to_thread(
                runner,
                lambda completed, total, text=None: loop.call_soon_threadsafe(
                    self._update_progress,
                    job.job_id,
                    completed,
                    total,
                    text or "",
                ),
            )
            job.result = _serialize(result)
            job.status = "succeeded"
            if job.progress_total > 0 and job.progress_completed < job.progress_total:
                job.progress_completed = job.progress_total
                job.progress_percent = 100.0
        except Exception as exc:
            job.error = str(exc)
            job.status = "failed"
            logger.error("后台任务执行失败：%s：%s", job.kind, exc)
        finally:
            job.finished_at = _now_text()
            async with self._lock:
                self._active_job_ids.discard(job.job_id)
                self._publish_locked()

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        self._push_queue(queue, self._snapshot())
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._subscribers.discard(queue)

    def _update_progress(self, job_id: str, completed: int, total: int, text: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return

        job.progress_completed = max(0, completed)
        job.progress_total = max(0, total)
        job.progress_text = text
        if job.progress_total > 0:
            job.progress_percent = min(100.0, job.progress_completed / job.progress_total * 100.0)
        else:
            job.progress_percent = 0.0
        self._publish()

    def _snapshot(self) -> str:
        return json.dumps({"jobs": self.list()}, ensure_ascii=False)

    def _publish(self) -> None:
        payload = self._snapshot()
        for queue in tuple(self._subscribers):
            self._push_queue(queue, payload)

    def _publish_locked(self) -> None:
        self._publish()

    @staticmethod
    def _push_queue(queue: asyncio.Queue[str], payload: str) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def get(self, job_id: str) -> JobRecord:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def list(self) -> list[dict[str, Any]]:
        jobs = sorted(
            self._jobs.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )
        return [job.to_dict() for job in jobs]


def create_app(config: ProjectConfig | None = None):
    from aiohttp import web

    cfg = config or ProjectConfig()
    app = web.Application()
    app["config"] = cfg
    app["jobs"] = JobManager(single_job_mode=cfg.web.single_job_mode)

    app.router.add_get("/", _index_handler)
    app.router.add_get("/api/config", _config_handler)
    app.router.add_get("/api/datasets", _datasets_handler)
    app.router.add_get("/api/datasets/{dataset_name}/images", _dataset_images_handler)
    app.router.add_get("/api/datasets/{dataset_name}/files", _dataset_files_handler)
    app.router.add_get("/api/jobs", _jobs_handler)
    app.router.add_get("/api/jobs/{job_id}", _job_detail_handler)
    app.router.add_get("/api/events/jobs", _jobs_events_handler)
    app.router.add_get("/api/artifacts", _artifacts_handler)
    app.router.add_get("/api/analysis/options", _analysis_options_handler)
    app.router.add_get("/api/file", _file_handler)
    app.router.add_get("/static/{path:.*}", _static_file_handler)

    app.router.add_post("/api/jobs/mask", _mask_job_handler)
    app.router.add_post("/api/jobs/auto-label", _auto_label_job_handler)
    app.router.add_post("/api/jobs/infer-dataset", _infer_dataset_job_handler)
    app.router.add_post("/api/jobs/train", _train_job_handler)
    app.router.add_post("/api/jobs/export-onnx", _export_job_handler)
    app.router.add_post("/api/jobs/export-trt", _export_trt_job_handler)
    app.router.add_post("/api/jobs/verify-onnx", _verify_job_handler)
    app.router.add_post("/api/jobs/infer-trt", _trt_job_handler)
    app.router.add_post("/api/jobs/navigate-onnx", _navigate_onnx_job_handler)
    app.router.add_post("/api/jobs/navigate-trt", _navigate_trt_job_handler)
    app.router.add_post("/api/jobs/navigate-dataset", _navigate_dataset_job_handler)
    app.router.add_post("/api/jobs/benchmark", _benchmark_job_handler)
    app.router.add_post("/api/jobs/export-paper-figures", _export_paper_figures_job_handler)
    app.router.add_post("/api/jobs/audit-navigation", _audit_navigation_job_handler)

    return app


def run_web_app(
    config: ProjectConfig | None = None,
    host: str | None = None,
    port: int | None = None,
    open_browser: bool = False,
) -> None:
    from aiohttp import web

    cfg = config or ProjectConfig()
    app = create_app(cfg)
    bind_host = host or cfg.web.host
    bind_port = port or cfg.web.port
    page_url = f"http://{bind_host}:{bind_port}"

    if open_browser:
        Timer(0.8, lambda: webbrowser.open(page_url)).start()

    logger.info("启动 Web 操作台：%s", page_url)
    web.run_app(app, host=bind_host, port=bind_port)


async def _index_handler(request):
    from aiohttp import web

    text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    response = web.Response(text=text, content_type="text/html", charset="utf-8")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


async def _config_handler(request):
    from aiohttp import web

    config = request.app["config"]
    return web.json_response(_serialize(asdict(config)))


async def _jobs_handler(request):
    from aiohttp import web

    jobs: JobManager = request.app["jobs"]
    return web.json_response({"jobs": jobs.list()})


async def _datasets_handler(request):
    from aiohttp import web

    config: ProjectConfig = request.app["config"]
    return web.json_response(_list_datasets(config))


async def _dataset_images_handler(request):
    from aiohttp import web

    config: ProjectConfig = request.app["config"]
    dataset_name = request.match_info["dataset_name"]
    datasets = {item["name"]: item for item in _list_datasets(config)["datasets"]}
    dataset = datasets.get(dataset_name)
    if dataset is None:
        raise web.HTTPNotFound(text=f"数据集不存在：{dataset_name}")
    return web.json_response(
        {
            "dataset_name": dataset_name,
            "image_dir": dataset.get("image_dir"),
            "images": _list_dataset_files(dataset, "images", config),
        }
    )


async def _dataset_files_handler(request):
    from aiohttp import web

    config: ProjectConfig = request.app["config"]
    dataset_name = request.match_info["dataset_name"]
    kind = str(request.query.get("kind", "images")).strip().lower() or "images"
    datasets = {item["name"]: item for item in _list_datasets(config)["datasets"]}
    dataset = datasets.get(dataset_name)
    if dataset is None:
        raise web.HTTPNotFound(text=f"数据集不存在：{dataset_name}")
    return web.json_response(
        {
            "dataset_name": dataset_name,
            "kind": kind,
            "files": _list_dataset_files(dataset, kind, config),
        }
    )


async def _job_detail_handler(request):
    from aiohttp import web

    jobs: JobManager = request.app["jobs"]
    job_id = request.match_info["job_id"]
    try:
        job = jobs.get(job_id)
    except KeyError as exc:
        raise web.HTTPNotFound(text=f"任务不存在：{job_id}") from exc
    return web.json_response(job.to_dict())


async def _jobs_events_handler(request):
    from aiohttp import web

    jobs: JobManager = request.app["jobs"]
    queue = jobs.subscribe()

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    try:
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=15)
                await response.write(f"data: {payload}\n\n".encode("utf-8"))
            except asyncio.TimeoutError:
                await response.write(b": ping\n\n")
    except (asyncio.CancelledError, ConnectionError):
        pass
    finally:
        jobs.unsubscribe(queue)
        with suppress(Exception):
            await response.write_eof()

    return response


async def _artifacts_handler(request):
    from aiohttp import web

    config: ProjectConfig = request.app["config"]
    artifacts = _list_artifacts(config.inference.artifacts_dir, config.web.recent_runs_limit)
    return web.json_response(artifacts)


async def _analysis_options_handler(request):
    from aiohttp import web

    config: ProjectConfig = request.app["config"]
    return web.json_response(_list_analysis_options(config))


async def _file_handler(request):
    from aiohttp import web

    raw_path = request.query.get("path", "").strip()
    if not raw_path:
        raise web.HTTPBadRequest(text="缺少文件路径。")

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate

    resolved = candidate.resolve()
    if not resolved.is_file():
        raise web.HTTPNotFound(text=f"文件不存在：{resolved}")
    if not resolved.is_relative_to(WORKSPACE_ROOT):
        raise web.HTTPForbidden(text="只允许访问工作区内文件。")

    return _serve_local_file(resolved, cache_control="no-store")


async def _static_file_handler(request):
    from aiohttp import web

    relative_path = str(request.match_info.get("path", "")).strip()
    if not relative_path:
        raise web.HTTPNotFound(text="静态文件不存在。")

    resolved = (STATIC_DIR / relative_path).resolve()
    if not resolved.is_file():
        raise web.HTTPNotFound(text=f"静态文件不存在：{relative_path}")
    if not resolved.is_relative_to(STATIC_DIR.resolve()):
        raise web.HTTPForbidden(text="禁止访问静态目录外文件。")

    return _serve_local_file(resolved, cache_control="public, max-age=3600")


async def _mask_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(request, "mask", payload, lambda progress: _run_mask(payload, progress))


async def _auto_label_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(request, "auto-label", payload, lambda progress: _run_auto_label(payload, progress))


async def _infer_dataset_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(
        request,
        "infer-dataset",
        payload,
        lambda progress: _run_infer_dataset(payload, progress),
    )


async def _train_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(request, "train", payload, lambda progress: _run_train(payload))


async def _export_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(request, "export-onnx", payload, lambda progress: _run_export_onnx(payload))


async def _export_trt_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(request, "export-trt", payload, lambda progress: _run_export_trt(payload))


async def _verify_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(request, "verify-onnx", payload, lambda progress: _run_verify_onnx(payload))


async def _trt_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(request, "infer-trt", payload, lambda progress: _run_infer_trt(payload))


async def _navigate_onnx_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(
        request,
        "navigate-onnx",
        payload,
        lambda progress: _run_navigate_onnx(payload),
    )


async def _navigate_trt_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(
        request,
        "navigate-trt",
        payload,
        lambda progress: _run_navigate_trt(payload),
    )


async def _navigate_dataset_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(
        request,
        "navigate-dataset",
        payload,
        lambda progress: _run_navigate_dataset(payload, progress),
    )


async def _benchmark_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(request, "benchmark", payload, lambda progress: _run_benchmark(payload))


async def _export_paper_figures_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(
        request,
        "export-paper-figures",
        payload,
        lambda progress: _run_export_paper_figures(payload),
    )


async def _audit_navigation_job_handler(request):
    payload = await _read_json(request)
    return await _submit_job(
        request,
        "audit-navigation",
        payload,
        lambda progress: _run_audit_navigation(payload),
    )


async def _submit_job(
    request,
    kind: str,
    payload: dict[str, Any],
    runner: Callable[[Callable[[int, int, str | None], None]], dict[str, Any]],
):
    from aiohttp import web

    jobs: JobManager = request.app["jobs"]
    try:
        job = await jobs.submit(kind, payload, runner)
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc)) from exc
    return web.json_response(job.to_dict(), status=202)


async def _read_json(request) -> dict[str, Any]:
    from aiohttp import web

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text="请求体不是合法 JSON。") from exc

    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="请求体必须是 JSON 对象。")
    return payload


def _run_mask(
    payload: dict[str, Any],
    progress_callback: Callable[[int, int, str | None], None] | None = None,
) -> dict[str, Any]:
    project = _build_project(payload)
    input_dir = _required_path(payload, "input_dir")
    output_dir = _optional_path(payload.get("output_dir"))
    outputs = project.generate_masks(input_dir, output_dir, progress_callback=progress_callback)
    return {
        "output_count": len(outputs),
        "outputs": [path.resolve() for path in outputs],
    }


def _run_auto_label(
    payload: dict[str, Any],
    progress_callback: Callable[[int, int, str | None], None] | None = None,
) -> dict[str, Any]:
    project = _build_project(payload)
    image_dir = _required_path(payload, "image_dir")
    annotation_dir = _optional_path(payload.get("annotation_dir"))
    overlay_dir = _optional_path(payload.get("overlay_dir"))
    return project.auto_label(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        overlay_dir=overlay_dir,
        progress_callback=progress_callback,
    )


def _run_infer_dataset(
    payload: dict[str, Any],
    progress_callback: Callable[[int, int, str | None], None] | None = None,
) -> dict[str, Any]:
    project = _build_project(payload)
    image_dir = _required_path(payload, "image_dir")
    overlay_dir = _optional_path(payload.get("overlay_dir"))
    return project.infer_dataset_onnx(
        image_dir=image_dir,
        overlay_dir=overlay_dir,
        progress_callback=progress_callback,
    )


def _run_train(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    image_dir = _required_path(payload, "image_dir")
    mask_dir = _required_path(payload, "mask_dir")
    _, history = project.train(image_dir, mask_dir)

    output_dir = project.training_service.output_dir.resolve()
    result = {
        "output_dir": output_dir,
        "history": history,
        "history_path": output_dir / "history.json",
        "best_model_path": output_dir / "best_model.pt",
    }

    onnx_path = output_dir / "model.onnx"
    if onnx_path.exists():
        result["onnx_path"] = onnx_path
    return result


def _run_export_onnx(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    output_path = _optional_path(payload.get("output_path"))
    exported = project.export_onnx(output_path)
    return {"onnx_path": exported.resolve()}


def _run_export_trt(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    onnx_path = _optional_path(payload.get("onnx_path"))
    output_path = _optional_path(payload.get("output_path"))
    exported = project.export_tensorrt(
        onnx_path=onnx_path,
        engine_output_path=output_path,
    )
    return {
        "onnx_path": (onnx_path.resolve() if onnx_path is not None else project.trt_inference_service._resolve_onnx_path().resolve()),
        "engine_path": exported,
    }


def _run_verify_onnx(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    image_path = _required_path(payload, "image_path")
    mask_output, overlay_output = project.verify_onnx(image_path)
    return {
        "image_path": image_path.resolve(),
        "mask_output": mask_output.resolve(),
        "overlay_output": overlay_output.resolve(),
    }


def _run_infer_trt(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    image_path = _required_path(payload, "image_path")
    mask_output, overlay_output = project.infer_tensorrt(image_path)
    return {
        "image_path": image_path.resolve(),
        "mask_output": mask_output.resolve(),
        "overlay_output": overlay_output.resolve(),
    }


def _run_navigate_onnx(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    image_path = _required_path(payload, "image_path")
    return project.navigate_onnx(image_path)


def _run_navigate_trt(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    image_path = _required_path(payload, "image_path")
    return project.navigate_tensorrt(image_path)


def _run_navigate_dataset(
    payload: dict[str, Any],
    progress_callback: Callable[[int, int, str | None], None] | None = None,
) -> dict[str, Any]:
    project = _build_project(payload)
    image_dir = _required_path(payload, "image_dir")
    return project.navigate_dataset_onnx(
        image_dir=image_dir,
        progress_callback=progress_callback,
    )


def _run_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    image_dir = _required_path(payload, "image_dir")
    output_dir = _optional_path(payload.get("output_dir"))
    backends = _optional_string_list(payload.get("backends"))
    return project.benchmark_inference(
        image_dir=image_dir,
        output_dir=output_dir,
        backends=backends or None,
        warmup_runs=_optional_int(payload.get("warmup_runs")),
        timed_runs=_optional_int(payload.get("timed_runs")),
        max_images=_optional_int(payload.get("max_images")),
    )


def _run_export_paper_figures(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    return project.export_paper_figures(
        history_path=_optional_path(payload.get("history_path")),
        navigation_root=_optional_path(payload.get("navigation_root")),
        output_dir=_optional_path(payload.get("output_dir")),
    )


def _run_audit_navigation(payload: dict[str, Any]) -> dict[str, Any]:
    project = _build_project(payload)
    return project.audit_navigation_results(
        navigation_root=_optional_path(payload.get("navigation_root")),
        output_dir=_optional_path(payload.get("output_dir")),
    )


def _build_project(payload: dict[str, Any]) -> SegmentationProject:
    config = ProjectConfig()
    overrides = payload.get("config", {})
    if isinstance(overrides, dict):
        _apply_overrides(config, overrides)
    return SegmentationProject(config)


def _apply_overrides(target: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            continue

        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
            continue

        setattr(target, key, _coerce_value(current, value))


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, Path):
        return Path(str(value)).expanduser()
    if current is None and isinstance(value, str):
        text = value.strip()
        return Path(text).expanduser() if text else None
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _required_path(payload: dict[str, Any], key: str) -> Path:
    raw = str(payload.get(key, "")).strip()
    if not raw:
        raise ValueError(f"缺少参数：{key}")
    return Path(raw).expanduser()


def _optional_path(value: Any) -> Path | None:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return None
    return Path(raw).expanduser()


def _required_text(payload: dict[str, Any], key: str) -> str:
    text = str(payload.get(key, "")).strip()
    if not text:
        raise ValueError(f"缺少参数：{key}")
    return text


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _list_artifacts(root_dir: Path, limit: int) -> dict[str, Any]:
    root_dir = root_dir.resolve()
    latest_path: Path | None = None
    try:
        latest_path = latest_run_dir(root_dir).resolve()
    except FileNotFoundError:
        latest_path = None

    root_files: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    if root_dir.exists():
        for item in sorted(root_dir.iterdir(), key=lambda path: path.name.lower()):
            if not item.is_file():
                continue
            root_files.append(
                {
                    "name": item.name,
                    "path": str(item.resolve()),
                    "size": item.stat().st_size,
                    "modified_at": _file_time(item),
                    "is_image": item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"},
                }
            )

        run_dirs = sorted(
            [path for path in root_dir.iterdir() if path.is_dir()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs[:limit]:
            files = []
            for item in sorted(run_dir.iterdir(), key=lambda path: path.name.lower()):
                if not item.is_file():
                    continue
                files.append(
                    {
                        "name": item.name,
                        "path": str(item.resolve()),
                        "size": item.stat().st_size,
                        "modified_at": _file_time(item),
                        "is_image": item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"},
                    }
                )
            runs.append(
                {
                    "name": run_dir.name,
                    "path": str(run_dir.resolve()),
                    "is_latest": latest_path == run_dir.resolve(),
                    "files": files,
                }
            )

    return {
        "root_dir": str(root_dir),
        "latest_run": str(latest_path) if latest_path else None,
        "root_files": root_files,
        "runs": runs,
    }


def _list_analysis_options(config: ProjectConfig) -> dict[str, Any]:
    train_root = config.train.output_dir.resolve()

    latest_path: Path | None = None
    try:
        latest_path = latest_run_dir(train_root).resolve()
    except FileNotFoundError:
        latest_path = None

    train_runs: list[dict[str, Any]] = []
    if train_root.exists():
        run_dirs = sorted(
            [path for path in train_root.iterdir() if path.is_dir() and path.name.startswith("run-")],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs:
            history_path = run_dir / "history.json"
            if not history_path.exists():
                continue
            train_runs.append(
                {
                    "name": run_dir.name,
                    "path": str(run_dir.resolve()),
                    "history_path": str(history_path.resolve()),
                    "best_model_path": str((run_dir / "best_model.pt").resolve()) if (run_dir / "best_model.pt").exists() else None,
                    "onnx_path": str((run_dir / "model.onnx").resolve()) if (run_dir / "model.onnx").exists() else None,
                    "trt_path": str((run_dir / "model.trt").resolve()) if (run_dir / "model.trt").exists() else None,
                    "is_latest": latest_path == run_dir.resolve(),
                    "modified_at": _file_time(run_dir),
                }
            )

    return {
        "train_root": str(train_root),
        "train_runs": train_runs,
    }


def _list_datasets(config: ProjectConfig) -> dict[str, Any]:
    sample_root = config.dataset.root_dir.resolve()
    datasets: list[dict[str, Any]] = []
    overlay_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    for path in iter_dataset_dirs(sample_root):
        dataset_paths = build_dataset_paths(config, path)
        for directory in (
            dataset_paths.image_dir,
            dataset_paths.annotation_dir,
            dataset_paths.mask_dir,
            dataset_paths.overlay_dir,
            dataset_paths.navigation_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        image_files = _collect_image_files(dataset_paths.image_dir, config.train.image_extensions)
        annotation_files = _collect_annotation_files(dataset_paths.annotation_dir)
        mask_files = _collect_mask_files(dataset_paths.mask_dir, config.train.mask_suffix)
        overlay_files = _collect_image_files(dataset_paths.overlay_dir, overlay_extensions)
        navigation_files = _collect_navigation_files(dataset_paths.navigation_dir)

        image_stems = {path.stem for path in image_files}
        annotation_stems = {path.stem for path in annotation_files}
        mask_stems = {_strip_suffix(path.name, config.train.mask_suffix) for path in mask_files}
        overlay_stems = _collect_stems_by_suffix(overlay_files, "_onnx_overlay")
        navigation_stems = _collect_stems_by_suffix(
            [path for path in navigation_files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}],
            "_onnx_nav_overlay",
            "_trt_nav_overlay",
        )

        image_count = len(image_files)
        annotation_count = len(image_stems & annotation_stems)
        mask_count = len(image_stems & mask_stems)
        overlay_count = len(image_stems & overlay_stems)
        navigation_count = len(image_stems & navigation_stems)

        datasets.append(
            {
                "name": dataset_paths.name,
                "path": str(dataset_paths.root_dir.resolve()),
                "image_dir": str(dataset_paths.image_dir.resolve()),
                "mask_dir": str(dataset_paths.mask_dir.resolve()),
                "annotation_dir": str(dataset_paths.annotation_dir.resolve()),
                "overlay_dir": str(dataset_paths.overlay_dir.resolve()),
                "navigation_dir": str(dataset_paths.navigation_dir.resolve()),
                "image_count": image_count,
                "annotation_count": annotation_count,
                "mask_count": mask_count,
                "overlay_count": overlay_count,
                "navigation_count": navigation_count,
                "has_images": image_count > 0,
                "has_masks": mask_count > 0,
                "has_annotations": annotation_count > 0,
                "has_overlays": overlay_count > 0,
                "has_navigation": navigation_count > 0,
            }
        )

    return {
        "root_dir": str(sample_root),
        "datasets": datasets,
    }


def _list_dataset_files(
    dataset: dict[str, Any],
    kind: str,
    config: ProjectConfig,
) -> list[dict[str, Any]]:
    directory_map = {
        "images": ("image_dir", _collect_image_files, config.train.image_extensions),
        "annotations": ("annotation_dir", _collect_annotation_files, None),
        "masks": ("mask_dir", _collect_mask_files, config.train.mask_suffix),
        "overlays": ("overlay_dir", _collect_image_files, (".png", ".jpg", ".jpeg", ".bmp")),
        "navigation": ("navigation_dir", _collect_navigation_files, None),
    }
    mapping = directory_map.get(kind)
    if mapping is None:
        return []

    directory_key, collector, collector_arg = mapping
    directory_value = dataset.get(directory_key)
    if not directory_value:
        return []

    root_dir = Path(directory_value)
    if collector_arg is None:
        files = collector(root_dir)  # type: ignore[misc]
    else:
        files = collector(root_dir, collector_arg)  # type: ignore[misc]

    return [
        {
            "name": path.name,
            "path": str(path.resolve()),
            "relative_path": path.name,
            "is_image": path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"},
        }
        for path in files
    ]

def _collect_image_files(root_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not root_dir.is_dir():
        return []

    normalized = {item.lower() for item in extensions}
    return sorted(
        [path for path in root_dir.iterdir() if path.is_file() and path.suffix.lower() in normalized],
        key=lambda item: str(item).lower(),
    )


def _collect_annotation_files(root_dir: Path) -> list[Path]:
    if not root_dir.is_dir():
        return []

    return sorted(
        [path for path in root_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"],
        key=lambda item: str(item).lower(),
    )


def _collect_mask_files(root_dir: Path, mask_suffix: str) -> list[Path]:
    if not root_dir.is_dir():
        return []

    normalized_suffix = mask_suffix.lower()
    return sorted(
        [path for path in root_dir.iterdir() if path.is_file() and path.name.lower().endswith(normalized_suffix)],
        key=lambda item: str(item).lower(),
    )


def _collect_navigation_files(root_dir: Path) -> list[Path]:
    if not root_dir.is_dir():
        return []

    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".json"}
    return sorted(
        [path for path in root_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed],
        key=lambda item: str(item).lower(),
    )


def _strip_suffix(filename: str, suffix: str) -> str:
    if filename.lower().endswith(suffix.lower()):
        return filename[: -len(suffix)]
    return Path(filename).stem


def _collect_stems_by_suffix(paths: list[Path], *suffixes: str) -> set[str]:
    normalized_suffixes = [suffix.lower() for suffix in suffixes]
    stems: set[str] = set()
    for path in paths:
        stem = path.stem
        for suffix, normalized_suffix in zip(suffixes, normalized_suffixes):
            if stem.lower().endswith(normalized_suffix):
                stems.add(stem[: -len(suffix)])
                break
    return stems


def _file_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _serve_local_file(path: Path, cache_control: str | None = None):
    from aiohttp import web

    content_type, encoding = mimetypes.guess_type(str(path))

    if content_type and (
        content_type.startswith("text/")
        or content_type in {"application/javascript", "application/json", "image/svg+xml"}
    ):
        text = path.read_text(encoding="utf-8")
        response = web.Response(text=text, content_type=content_type, charset="utf-8")
    else:
        body = path.read_bytes()
        response = web.Response(body=body)
        if content_type:
            response.content_type = content_type

    if encoding:
        response.headers["Content-Encoding"] = encoding
    if cache_control:
        response.headers["Cache-Control"] = cache_control
    return response
