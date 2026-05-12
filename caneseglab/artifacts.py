from __future__ import annotations

from datetime import datetime
from pathlib import Path

LATEST_RUN_FILE = "latest.txt"


def create_run_dir(root_dir: Path) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = root_dir / base_name
    suffix = 1

    while run_dir.exists():
        suffix += 1
        run_dir = root_dir / f"{base_name}-{suffix}"

    run_dir.mkdir(parents=True, exist_ok=False)
    (root_dir / LATEST_RUN_FILE).write_text(run_dir.name, encoding="utf-8")
    return run_dir


def latest_run_dir(root_dir: Path) -> Path:
    if not root_dir.exists():
        raise FileNotFoundError(f"训练输出根目录不存在：{root_dir}")

    marker = root_dir / LATEST_RUN_FILE
    if marker.exists():
        name = marker.read_text(encoding="utf-8").strip()
        if name:
            candidate = root_dir / name
            if candidate.is_dir():
                return candidate

    run_dirs = [path for path in root_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"未找到训练输出目录：{root_dir}")

    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def resolve_artifact_path(
    root_dir: Path,
    filename: str,
    explicit_path: Path | None = None,
) -> Path:
    if explicit_path is not None:
        return explicit_path
    legacy_path = root_dir / filename
    if legacy_path.exists():
        return legacy_path
    return latest_run_dir(root_dir) / filename
