from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..config import ProjectConfig
from .onnx_inference_service import OnnxInferenceService


class DatasetInferenceService:
    """批量补齐数据集中的 ONNX 推理结果。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.project_cfg = config
        self.inference_service = OnnxInferenceService(config)

    def infer_missing_overlays(
        self,
        image_dir: Path,
        overlay_dir: Path | None = None,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> dict[str, Any]:
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)

        overlay_dir = overlay_dir or image_dir.parent / self.project_cfg.dataset.overlay_dirname
        overlay_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {item.lower() for item in self.project_cfg.train.image_extensions}
        image_paths = sorted(
            [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_extensions],
            key=lambda item: item.name.lower(),
        )

        pending_images = [
            image_path
            for image_path in image_paths
            if not (overlay_dir / f"{image_path.stem}_onnx_overlay.png").exists()
        ]

        if progress_callback is not None:
            progress_callback(0, len(pending_images), "无需处理" if not pending_images else "准备推理")

        overlay_outputs: list[Path] = []
        mask_outputs: list[Path] = []

        for index, image_path in enumerate(pending_images, start=1):
            mask_output, overlay_output = self.inference_service.infer_file(image_path)
            mask_outputs.append(mask_output)
            overlay_outputs.append(overlay_output)
            if progress_callback is not None:
                progress_callback(index, len(pending_images), image_path.name)

        return {
            "image_count": len(image_paths),
            "processed_count": len(pending_images),
            "skipped_existing": len(image_paths) - len(pending_images),
            "mask_outputs": [path.resolve() for path in mask_outputs],
            "overlay_outputs": [path.resolve() for path in overlay_outputs],
        }
