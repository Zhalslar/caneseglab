from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig


@dataclass(frozen=True)
class DatasetPaths:
    """标准数据集目录。"""

    name: str
    root_dir: Path
    image_dir: Path
    annotation_dir: Path
    mask_dir: Path
    overlay_dir: Path
    navigation_dir: Path


def build_dataset_paths(config: ProjectConfig, dataset_dir: Path) -> DatasetPaths:
    """按统一规范构造数据集目录路径。"""
    return DatasetPaths(
        name=dataset_dir.name,
        root_dir=dataset_dir,
        image_dir=dataset_dir / config.dataset.images_dirname,
        annotation_dir=dataset_dir / config.dataset.annotations_dirname,
        mask_dir=dataset_dir / config.dataset.masks_dirname,
        overlay_dir=dataset_dir / config.dataset.overlay_dirname,
        navigation_dir=dataset_dir / config.dataset.navigation_dirname,
    )


def iter_dataset_dirs(root_dir: Path) -> list[Path]:
    """列出数据集目录。"""
    if not root_dir.exists():
        return []

    return sorted(
        [path for path in root_dir.iterdir() if path.is_dir()],
        key=lambda item: item.name.lower(),
    )


def find_dataset_paths_for_image(
    config: ProjectConfig,
    image_path: Path,
) -> DatasetPaths | None:
    """根据图片路径反推所属数据集目录。"""
    for parent in image_path.parents:
        if parent.name != config.dataset.images_dirname:
            continue
        return build_dataset_paths(config, parent.parent)
    return None
