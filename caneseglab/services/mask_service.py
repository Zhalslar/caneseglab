from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
import numpy.typing as npt

from ..config import ProjectConfig
from .log_service import logger


class LabelmeMaskGenerator:
    """Labelme JSON 转分割掩码服务。"""

    def __init__(self, config: ProjectConfig):
        self.project_cfg = config
        self.cfg = config.mask

    def generate(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> list[Path]:
        """生成 mask"""
        if not input_dir.exists():
            raise FileNotFoundError(input_dir)

        if output_dir is None:
            output_dir = input_dir.parent / self.project_cfg.dataset.masks_dirname

        output_dir.mkdir(parents=True, exist_ok=True)

        json_paths = sorted(input_dir.glob(self.cfg.json_pattern), key=lambda item: item.name.lower())
        pending_paths = [
            json_path
            for json_path in json_paths
            if not (output_dir / f"{json_path.stem}{self.cfg.output_suffix}").exists()
        ]

        if progress_callback is not None:
            progress_callback(0, len(pending_paths), "无需处理" if not pending_paths else "准备生成掩码")

        outputs: list[Path] = []

        for index, json_path in enumerate(pending_paths, start=1):
            try:
                mask = self._build_mask(json_path)

                out_path = output_dir / f"{json_path.stem}{self.cfg.output_suffix}"

                cv2.imwrite(str(out_path), mask)

                outputs.append(out_path)

                logger.info("掩码生成完成：%s", json_path.name)
            except Exception as e:
                logger.error("掩码生成失败：%s：%s", json_path.name, e)
            finally:
                if progress_callback is not None:
                    progress_callback(index, len(pending_paths), json_path.name)

        return outputs

    def _build_mask(self, json_path: Path) -> npt.NDArray[np.uint8]:
        """构建 mask"""
        with json_path.open("r", encoding="utf8") as f:
            data = json.load(f)

        width: int = int(data["imageWidth"])
        height: int = int(data["imageHeight"])

        mask: npt.NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)

        polygons: List[npt.NDArray[np.int32]] = [
            np.asarray(shape["points"], dtype=np.int32)
            for shape in data["shapes"]
            if shape.get("shape_type") == "polygon"
        ]

        if polygons:
            cv2.fillPoly(mask, polygons, self.cfg.mask_value) # type: ignore

        return mask
