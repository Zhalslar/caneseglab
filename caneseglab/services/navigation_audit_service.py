from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from ..datasets import build_dataset_paths


class NavigationAuditService:
    """导航结果审计服务。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.project_cfg = config

    def audit(
        self,
        navigation_root: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        root = (navigation_root or self.project_cfg.dataset.root_dir).resolve()
        nav_paths, scope = self._collect_navigation_json(root)
        if not nav_paths:
            raise RuntimeError(f"未找到导航 JSON: {root}")

        status_counter: Counter[str] = Counter()
        dataset_counter: dict[str, Counter[str]] = defaultdict(Counter)
        failed_cases: list[dict[str, Any]] = []

        for path in nav_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = payload.get("result", {})
            dataset_name = path.parent.parent.name
            status = str(result.get("status", "unknown"))
            status_counter[status] += 1
            dataset_counter[dataset_name][status] += 1

            if status != "ok":
                failed_cases.append(
                    {
                        "dataset_name": dataset_name,
                        "file_name": path.name,
                        "message": result.get("message"),
                        "foreground_ratio": result.get("foreground_ratio"),
                        "point_count": result.get("point_count"),
                        "image_path": payload.get("image_path"),
                        "mask_path": payload.get("mask_path"),
                        "nav_json_path": str(path.resolve()),
                    }
                )

        export_dir = (output_dir or root).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        json_path = export_dir / "navigation_audit.json"
        csv_path = export_dir / "navigation_failures.csv"

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "navigation_root": str(root),
            "scope": scope,
            "total_nav_json": len(nav_paths),
            "status_counts": dict(status_counter),
            "per_dataset_status": {
                name: dict(counter)
                for name, counter in sorted(dataset_counter.items(), key=lambda item: item[0])
            },
            "failed_cases": failed_cases,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_failure_csv(csv_path, failed_cases)

        payload["summary_path"] = str(json_path.resolve())
        payload["failure_table_path"] = str(csv_path.resolve())
        return payload

    def _collect_navigation_json(self, root: Path) -> tuple[list[Path], str]:
        dataset_paths = build_dataset_paths(self.project_cfg, root)
        dataset_navigation_dir = dataset_paths.navigation_dir
        if dataset_navigation_dir.is_dir():
            nav_paths = sorted(
                dataset_navigation_dir.glob("*_nav.json"),
                key=lambda item: str(item).lower(),
            )
            if nav_paths:
                return nav_paths, "dataset"

        nav_paths = sorted(
            root.glob(f"*/{self.project_cfg.dataset.navigation_dirname}/*_nav.json"),
            key=lambda item: str(item).lower(),
        )
        return nav_paths, "root"

    @staticmethod
    def _write_failure_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "dataset_name",
            "file_name",
            "message",
            "foreground_ratio",
            "point_count",
            "image_path",
            "mask_path",
            "nav_json_path",
        ]
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
