from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from ..config import ProjectConfig
from .log_service import logger
from .onnx_inference_service import OnnxInferenceService
from .pytorch_inference_service import PytorchInferenceService


@dataclass(frozen=True)
class BenchmarkImage:
    """测速输入图像。"""

    path: Path
    width: int
    height: int


class BenchmarkService:
    """推理测速服务。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.project_cfg = config
        self.cfg = config.benchmark

    def benchmark(
        self,
        image_dir: Path,
        output_dir: Path | None = None,
        backends: list[str] | None = None,
        warmup_runs: int | None = None,
        timed_runs: int | None = None,
        max_images: int | None = None,
    ) -> dict[str, Any]:
        images = self._collect_images(image_dir, max_images or self.cfg.max_images)
        if not images:
            raise RuntimeError(f"测速目录中未找到可用图片: {image_dir}")

        export_dir = self._resolve_output_dir(output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        backend_list = backends or list(self.cfg.backends)
        warmup = max(0, warmup_runs if warmup_runs is not None else self.cfg.warmup_runs)
        timed = max(1, timed_runs if timed_runs is not None else self.cfg.timed_runs)

        results = []
        for backend in backend_list:
            logger.info("开始测速后端：%s", backend)
            result = self._benchmark_backend(backend, images, warmup, timed)
            results.append(result)
            logger.info(
                "测速后端完成：%s status=%s avg_ms=%s fps=%s",
                backend,
                result.get("status"),
                result.get("avg_ms"),
                result.get("fps"),
            )

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "image_dir": str(image_dir.resolve()),
            "image_count": len(images),
            "image_width_mean": round(mean(image.width for image in images), 3),
            "image_height_mean": round(mean(image.height for image in images), 3),
            "image_width_min": min(image.width for image in images),
            "image_width_max": max(image.width for image in images),
            "image_height_min": min(image.height for image in images),
            "image_height_max": max(image.height for image in images),
            "sample_images": [str(image.path.resolve()) for image in images],
            "warmup_runs": warmup,
            "timed_runs": timed,
            "timing_scope": "preprocess+inference+postprocess",
            "include_file_io": False,
            "environment": self._environment_snapshot(),
            "results": results,
        }

        json_path = export_dir / "benchmark_summary.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        csv_path = export_dir / "benchmark_table.csv"
        self._write_csv(csv_path, results)

        payload["summary_path"] = str(json_path.resolve())
        payload["table_path"] = str(csv_path.resolve())
        return payload

    def _benchmark_backend(
        self,
        backend: str,
        images: list[BenchmarkImage],
        warmup_runs: int,
        timed_runs: int,
    ) -> dict[str, Any]:
        name = backend.strip().lower()
        runner, metadata = self._build_runner(name)
        image_arrays = [cv2.imread(str(item.path), cv2.IMREAD_COLOR) for item in images]
        if any(image is None for image in image_arrays):
            raise RuntimeError(f"{backend} 测速时读取图片失败")

        def cycle_image(index: int):
            return image_arrays[index % len(image_arrays)]

        try:
            for index in range(warmup_runs):
                runner(cycle_image(index))
        except Exception as exc:
            return {
                "backend": backend,
                "status": "failed",
                "error": str(exc),
                "warmup_runs": warmup_runs,
                "timed_runs": timed_runs,
                **metadata,
            }

        durations_ms: list[float] = []
        try:
            for index in range(timed_runs):
                image = cycle_image(index)
                started = perf_counter()
                runner(image)
                durations_ms.append((perf_counter() - started) * 1000.0)
        except Exception as exc:
            return {
                "backend": backend,
                "status": "failed",
                "error": str(exc),
                "warmup_runs": warmup_runs,
                "timed_runs": len(durations_ms),
                **metadata,
            }

        sorted_durations = sorted(durations_ms)
        avg_ms = sum(durations_ms) / len(durations_ms)
        p50_ms = self._percentile(sorted_durations, 0.5)
        p95_ms = self._percentile(sorted_durations, 0.95)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        return {
            "backend": backend,
            "status": "ok",
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "avg_ms": round(avg_ms, 4),
            "p50_ms": round(p50_ms, 4),
            "p95_ms": round(p95_ms, 4),
            "min_ms": round(min(sorted_durations), 4),
            "max_ms": round(max(sorted_durations), 4),
            "fps": round(fps, 4),
            **metadata,
        }

    def _build_runner(self, backend: str):
        if backend == "pytorch":
            service = PytorchInferenceService(self.project_cfg)
            metadata = {
                "model_path": str(service.weight_path.resolve()),
                "device": str(service.device),
                "backend_detail": "torch",
            }
            return service.infer_image, metadata
        if backend == "onnx":
            service = OnnxInferenceService(self.project_cfg)
            metadata = {
                "model_path": str(service._onnx_path.resolve()) if service._onnx_path else None,
                "device": "onnxruntime",
                "backend_detail": ",".join(service._onnx_session.get_providers()) if service._onnx_session else None,
            }
            return service.infer_image, metadata
        if backend in {"tensorrt", "trt"}:
            from .tensorrt_inference_service import TensorRTInferenceService

            service = TensorRTInferenceService(self.project_cfg)
            try:
                engine_path = service._resolve_engine_path()
            except Exception:
                engine_path = None
            metadata = {
                "model_path": str(engine_path.resolve()) if engine_path and engine_path.exists() else None,
                "device": "tensorrt",
                "backend_detail": "tensorrt",
            }
            return service.infer_image, metadata
        raise ValueError(f"不支持的测速后端: {backend}")

    def _collect_images(self, image_dir: Path, max_images: int) -> list[BenchmarkImage]:
        image_extensions = {item.lower() for item in self.project_cfg.train.image_extensions}
        paths = sorted(
            [
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in image_extensions
            ],
            key=lambda item: item.name.lower(),
        )
        if max_images > 0:
            paths = paths[:max_images]

        images: list[BenchmarkImage] = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            images.append(
                BenchmarkImage(
                    path=path,
                    width=int(image.shape[1]),
                    height=int(image.shape[0]),
                )
            )
        return images

    def _resolve_output_dir(self, output_dir: Path | None) -> Path:
        root = output_dir or self.cfg.output_dir
        return root / datetime.now().strftime("bench-%Y%m%d-%H%M%S")

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "backend",
            "status",
            "warmup_runs",
            "timed_runs",
            "avg_ms",
            "p50_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
            "fps",
            "model_path",
            "device",
            "backend_detail",
            "error",
        ]
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _environment_snapshot() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "python": None,
            "torch": None,
            "cuda_available": None,
            "onnxruntime": None,
            "numpy": np.__version__,
        }
        try:
            import sys

            payload["python"] = sys.version.split()[0]
        except Exception:
            pass
        try:
            import torch

            payload["torch"] = torch.__version__
            payload["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                payload["cuda_device_count"] = int(torch.cuda.device_count())
                payload["cuda_device_name"] = torch.cuda.get_device_name(0)
                payload["torch_cuda"] = torch.version.cuda
        except Exception as exc:
            payload["torch_error"] = str(exc)
        try:
            import onnxruntime as ort

            payload["onnxruntime"] = ort.__version__
            payload["onnxruntime_providers"] = ort.get_available_providers()
        except Exception as exc:
            payload["onnxruntime_error"] = str(exc)
        return payload
