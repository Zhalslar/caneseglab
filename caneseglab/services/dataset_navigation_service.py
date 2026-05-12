from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..config import ProjectConfig
from ..datasets import build_dataset_paths
from .navigation_service import NavigationService
from .onnx_inference_service import OnnxInferenceService


class DatasetNavigationService:
    """批量生成数据集导航线结果。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.project_cfg = config
        self.inference_service = OnnxInferenceService(config)
        self.navigation_service = NavigationService(config)

    def navigate_missing_results(
        self,
        image_dir: Path,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> dict[str, Any]:
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)

        dataset_paths = build_dataset_paths(self.project_cfg, image_dir.parent)
        dataset_paths.navigation_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {item.lower() for item in self.project_cfg.train.image_extensions}
        image_paths = sorted(
            [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_extensions],
            key=lambda item: item.name.lower(),
        )

        pending_images = [
            image_path
            for image_path in image_paths
            if not (dataset_paths.navigation_dir / f"{image_path.stem}_onnx_nav_overlay.png").exists()
        ]

        if progress_callback is not None:
            progress_callback(0, len(pending_images), "无需处理" if not pending_images else "准备生成导航结果")

        mask_outputs: list[Path] = []
        overlay_outputs: list[Path] = []
        nav_overlay_outputs: list[Path] = []
        nav_json_outputs: list[Path] = []

        for index, image_path in enumerate(pending_images, start=1):
            mask_output, overlay_output = self.inference_service.infer_file(image_path)
            navigation_result = self.navigation_service.build_from_files(
                image_path=image_path,
                mask_path=mask_output,
                overlay_path=overlay_output,
                suffix="onnx",
            )
            mask_outputs.append(mask_output)
            overlay_outputs.append(overlay_output)
            nav_overlay_outputs.append(Path(str(navigation_result["nav_overlay_output"])))
            nav_json_outputs.append(Path(str(navigation_result["nav_json_output"])))
            if progress_callback is not None:
                progress_callback(index, len(pending_images), image_path.name)

        return {
            "image_count": len(image_paths),
            "processed_count": len(pending_images),
            "skipped_existing": len(image_paths) - len(pending_images),
            "mask_outputs": [path.resolve() for path in mask_outputs],
            "overlay_outputs": [path.resolve() for path in overlay_outputs],
            "nav_overlay_outputs": [path.resolve() for path in nav_overlay_outputs],
            "nav_json_outputs": [path.resolve() for path in nav_json_outputs],
        }
