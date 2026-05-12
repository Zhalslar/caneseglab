from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

    from .config import TrainConfig

Point = tuple[float, float]


@dataclass(frozen=True)
class Shape:
    label: str
    points: list[Point]
    group_id: int | None
    description: str
    shape_type: str
    flags: dict[str, Any]
    mask: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Shape | None:
        points_raw = raw.get("points")
        if not isinstance(points_raw, list) or not points_raw:
            return None

        points: list[Point] = []
        for point in points_raw:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                continue
            points.append((x, y))

        if not points:
            return None

        return cls(
            label=str(raw.get("label", "")),
            points=points,
            group_id=raw.get("group_id"),
            description=str(raw.get("description", "")),
            shape_type=str(raw.get("shape_type", "polygon")),
            flags=dict(raw.get("flags", {})),
            mask=raw.get("mask"),
        )


@dataclass(frozen=True)
class Annotation:
    version: str
    flags: dict[str, Any]
    shapes: list[Shape]
    image_path: str
    image_data: str | None
    image_height: int
    image_width: int

    @classmethod
    def from_file(cls, path: Path) -> Annotation:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw, source=path)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: Path | None = None) -> Annotation:
        width = int(raw.get("imageWidth", 0))
        height = int(raw.get("imageHeight", 0))
        if width <= 0 or height <= 0:
            origin = str(source) if source else "<memory>"
            raise ValueError(f"{origin} 图像尺寸非法: width={width}, height={height}")

        shapes: list[Shape] = []
        for item in raw.get("shapes", []):
            if not isinstance(item, dict):
                continue
            shape = Shape.from_dict(item)
            if shape is not None:
                shapes.append(shape)

        return cls(
            version=str(raw.get("version", "")),
            flags=dict(raw.get("flags", {})),
            shapes=shapes,
            image_path=str(raw.get("imagePath", "")),
            image_data=raw.get("imageData"),
            image_height=height,
            image_width=width,
        )

    def polygons(self) -> list[list[Point]]:
        polygons: list[list[Point]] = []

        for shape in self.shapes:
            polygon: list[Point] = []
            if shape.shape_type == "polygon":
                polygon = [
                    self._clip_point(x, y, self.image_width, self.image_height)
                    for x, y in shape.points
                ]
            elif shape.shape_type == "rectangle" and len(shape.points) >= 2:
                polygon = self._rectangle_to_polygon(
                    shape.points,
                    self.image_width,
                    self.image_height,
                )

            if self._is_valid_polygon(polygon):
                polygons.append(polygon)

        return polygons

    @staticmethod
    def _rectangle_to_polygon(
        points: list[Point],
        width: int,
        height: int,
    ) -> list[Point]:
        x1, y1 = points[0]
        x2, y2 = points[1]
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))

        return [
            Annotation._clip_point(left, top, width, height),
            Annotation._clip_point(right, top, width, height),
            Annotation._clip_point(right, bottom, width, height),
            Annotation._clip_point(left, bottom, width, height),
        ]

    @staticmethod
    def _clip_point(x: float, y: float, width: int, height: int) -> Point:
        max_x = max(0, width - 1)
        max_y = max(0, height - 1)
        clipped_x = min(max(float(x), 0.0), float(max_x))
        clipped_y = min(max(float(y), 0.0), float(max_y))
        return (clipped_x, clipped_y)

    @staticmethod
    def _is_valid_polygon(polygon: list[Point]) -> bool:
        if len(polygon) < 3:
            return False

        unique_points = {(round(x, 3), round(y, 3)) for x, y in polygon}
        return len(unique_points) >= 3


class SegmentationModel:
    """统一的分割模型构建入口。"""

    @staticmethod
    def build(
        model_name: str,
        backbone: str,
        in_channels: int,
        num_classes: int,
    ) -> "nn.Module":
        normalized = model_name.lower()

        if normalized == "originalunet":
            return build_original_unet(
                in_channels=in_channels,
                num_classes=num_classes,
            )

        if normalized != "unet":
            raise ValueError(f"暂不支持的模型类型: {model_name}")

        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    @classmethod
    def from_train_config(cls, config: "TrainConfig") -> "nn.Module":
        return cls.build(
            model_name=config.model_name,
            backbone=config.backbone,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
        )


def build_original_unet(
    in_channels: int,
    num_classes: int,
) -> "nn.Module":
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class Down(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(in_ch, out_ch),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class Up(nn.Module):
        def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = self.up(x)

            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            if diff_x != 0 or diff_y != 0:
                x = nn.functional.pad(
                    x,
                    [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
                )

            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    class OriginalUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inc = DoubleConv(in_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024)
            self.up1 = Up(1024, 512, 512)
            self.up2 = Up(512, 256, 256)
            self.up3 = Up(256, 128, 128)
            self.up4 = Up(128, 64, 64)
            self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            return self.outc(x)

    return OriginalUNet()
