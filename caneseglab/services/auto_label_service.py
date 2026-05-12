from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..config import ProjectConfig
from .log_service import logger
from .onnx_inference_service import OnnxInferenceService


class AutoLabelService:
    """基于已训练模型为未标注图片生成 Labelme 标注。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.project_cfg = config
        self.cfg = config.auto_label
        self.inference_service = OnnxInferenceService(config)

    def label_unannotated(
        self,
        image_dir: Path,
        annotation_dir: Path | None = None,
        overlay_dir: Path | None = None,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> dict[str, Any]:
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)

        annotation_dir = annotation_dir or image_dir.parent / self.project_cfg.dataset.annotations_dirname
        overlay_dir = overlay_dir or image_dir.parent / self.project_cfg.dataset.overlay_dirname
        annotation_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        created_annotations: list[Path] = []
        created_overlays: list[Path] = []
        skipped_existing = 0
        skipped_empty = 0

        image_extensions = {item.lower() for item in self.project_cfg.train.image_extensions}
        image_paths = sorted(
            [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_extensions],
            key=lambda item: item.name.lower(),
        )

        pending_images = []
        for image_path in image_paths:
            if (annotation_dir / f"{image_path.stem}.json").exists():
                skipped_existing += 1
                continue
            pending_images.append(image_path)

        if progress_callback is not None:
            progress_callback(0, len(pending_images), "无需处理" if not pending_images else "准备自动打标")

        for index, image_path in enumerate(pending_images, start=1):
            annotation_path = annotation_dir / f"{image_path.stem}.json"
            try:
                image = self.inference_service._load_image(image_path)
                mask, overlay = self.inference_service.infer_image(image)
                mask = self._restore_mask_size(mask, image.shape[:2])
                overlay = self._restore_overlay_size(overlay, image.shape[:2])

                shapes = self._build_shapes(mask)
                if not shapes and not self.cfg.write_empty_annotations:
                    skipped_empty += 1
                    continue

                annotation_data = self._build_labelme_annotation(
                    image_path=image_path,
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    shapes=shapes,
                )
                annotation_path.write_text(
                    json.dumps(annotation_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                created_annotations.append(annotation_path)
                logger.info("自动打标完成：%s", image_path.name)

                if self.cfg.save_overlay:
                    overlay_path = overlay_dir / f"{image_path.stem}{self.cfg.overlay_suffix}"
                    cv2.imwrite(str(overlay_path), overlay)
                    created_overlays.append(overlay_path)
            finally:
                if progress_callback is not None:
                    progress_callback(index, len(pending_images), image_path.name)

        return {
            "image_count": len(image_paths),
            "processed_count": len(pending_images),
            "created_count": len(created_annotations),
            "overlay_count": len(created_overlays),
            "skipped_existing": skipped_existing,
            "skipped_empty": skipped_empty,
            "annotations": [path.resolve() for path in created_annotations],
            "overlays": [path.resolve() for path in created_overlays],
        }

    def _build_shapes(self, mask: np.ndarray) -> list[dict[str, Any]]:
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes: list[dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.cfg.min_area:
                continue

            epsilon = cv2.arcLength(contour, True) * self.cfg.approx_epsilon_ratio
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            points = polygon.reshape(-1, 2).astype(float).tolist()
            if len(points) < 3:
                continue

            shapes.append(
                {
                    "label": self.cfg.label_name,
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                }
            )

        return shapes

    def _build_labelme_annotation(
        self,
        image_path: Path,
        image_height: int,
        image_width: int,
        shapes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "version": self.cfg.json_version,
            "flags": {"auto_label": True},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }

    @staticmethod
    def _restore_mask_size(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
        height, width = image_hw
        if mask.shape == (height, width):
            return mask
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _restore_overlay_size(overlay: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
        height, width = image_hw
        if overlay.shape[:2] == (height, width):
            return overlay
        return cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)
