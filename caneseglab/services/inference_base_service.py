from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import ProjectConfig
from ..datasets import find_dataset_paths_for_image


class InferenceBaseService:
    """ONNX / TensorRT 共享推理逻辑。"""

    def __init__(self, config: ProjectConfig) -> None:
        self._project_cfg = config
        self.cfg = config.inference
        self._overlay_color = np.array(self.cfg.overlay_color, dtype=np.uint8)

    # =========================
    # 图像处理
    # =========================

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"读取图像失败: {path}")
        return image

    def _preprocess(
        self,
        image: np.ndarray,
        input_hw: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """图像预处理。"""
        working = image

        target_width = self.cfg.resize_width
        target_height = self.cfg.resize_height
        if target_width is None and target_height is None and input_hw is not None:
            target_height, target_width = input_hw

        if target_width and target_height:
            working = cv2.resize(working, (target_width, target_height))

        tensor = working.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, working

    # =========================
    # 后处理
    # =========================

    def _logits_to_mask(self, logits: np.ndarray) -> np.ndarray:
        """模型输出转 mask。"""
        if logits.ndim != 4:
            raise ValueError(f"模型输出维度异常: {logits.shape}")

        logits = logits[0]

        # 二分类
        if logits.shape[0] == 1:
            prob = 1.0 / (1.0 + np.exp(-logits[0]))
            return (prob >= self.cfg.threshold).astype(np.uint8) * 255

        # 多分类
        class_map = np.argmax(logits, axis=0).astype(np.uint8)

        max_class = int(class_map.max()) if class_map.size else 0
        if max_class == 0:
            return class_map

        return (class_map.astype(np.float32) / max_class * 255).astype(np.uint8)

    def _build_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """mask 可视化叠加。"""
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        overlay = image.copy()
        foreground = np.empty_like(image)
        foreground[:] = self._overlay_color

        mask_bool = mask > 0
        overlay[mask_bool] = cv2.addWeighted(
            image[mask_bool],
            1 - self.cfg.overlay_alpha,
            foreground[mask_bool],
            self.cfg.overlay_alpha,
            0,
        )

        return overlay

    # =========================
    # 输出
    # =========================

    def _save_result(
        self,
        image_path: Path,
        mask: np.ndarray,
        overlay: np.ndarray,
        suffix: str,
    ) -> tuple[Path, Path]:
        dataset_paths = find_dataset_paths_for_image(self._project_cfg, image_path)

        if self.cfg.mask_output is not None:
            mask_output = self.cfg.mask_output
        elif dataset_paths is not None:
            mask_output = dataset_paths.mask_dir / f"{image_path.stem}_{suffix}_mask.png"
        else:
            mask_output = image_path.with_name(f"{image_path.stem}_{suffix}_mask.png")

        if self.cfg.overlay_output is not None:
            overlay_output = self.cfg.overlay_output
        elif dataset_paths is not None:
            overlay_output = dataset_paths.overlay_dir / f"{image_path.stem}_{suffix}_overlay.png"
        else:
            overlay_output = image_path.with_name(f"{image_path.stem}_{suffix}_overlay.png")

        mask_output.parent.mkdir(parents=True, exist_ok=True)
        overlay_output.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(mask_output), mask)
        cv2.imwrite(str(overlay_output), overlay)

        return mask_output, overlay_output
