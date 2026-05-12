from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import ProjectConfig
from ..datasets import find_dataset_paths_for_image


class NavigationService:
    """基于 BEV 的垄间导航线拟合服务。"""

    def __init__(self, config: ProjectConfig) -> None:
        self._project_cfg = config
        self.cfg = config.navigation

    def build_from_files(
        self,
        image_path: Path,
        mask_path: Path,
        overlay_path: Path | None,
        suffix: str,
    ) -> dict[str, Any]:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"读取导航掩膜失败: {mask_path}")

        result = self.fit_mask(mask)
        nav_overlay, bev_overlay = self._build_navigation_overlays(
            image_path=image_path,
            overlay_path=overlay_path,
            mask=mask,
            result=result,
        )
        nav_overlay_path, bev_overlay_path, nav_json_path = self._resolve_output_paths(image_path, suffix)

        nav_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(nav_overlay_path), nav_overlay)
        cv2.imwrite(str(bev_overlay_path), bev_overlay)

        payload = {
            "image_path": str(image_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "overlay_path": str(overlay_path.resolve()) if overlay_path else None,
            "nav_overlay_output": str(nav_overlay_path.resolve()),
            "bev_overlay_output": str(bev_overlay_path.resolve()),
            "nav_json_output": str(nav_json_path.resolve()),
            "result": result,
        }
        nav_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    def fit_mask(self, mask: np.ndarray) -> dict[str, Any]:
        binary = self._prepare_binary(mask)
        image_height, image_width = binary.shape
        foreground_ratio = float(binary.mean()) if binary.size else 0.0

        image_rows = self._collect_row_samples(binary)
        if len(image_rows) < self.cfg.min_points:
            return self._empty_result(
                message="原图中有效边界点不足，无法估计垄间区域。",
                image_hw=(image_height, image_width),
                foreground_ratio=foreground_ratio,
                image_rows=image_rows,
            )

        source_quad = self._estimate_source_quad(image_rows, image_width)
        if source_quad is None:
            return self._empty_result(
                message="无法从当前掩膜中稳定估计逆透视区域。",
                image_hw=(image_height, image_width),
                foreground_ratio=foreground_ratio,
                image_rows=image_rows,
            )

        bev_width = int(self.cfg.bev_width)
        bev_height = int(self.cfg.bev_height)
        destination_quad = self._destination_quad(bev_width, bev_height)
        perspective_matrix = cv2.getPerspectiveTransform(source_quad, destination_quad)
        inverse_matrix = cv2.getPerspectiveTransform(destination_quad, source_quad)

        bev_binary = cv2.warpPerspective(
            (binary * 255).astype(np.uint8),
            perspective_matrix,
            (bev_width, bev_height),
            flags=cv2.INTER_NEAREST,
        )
        bev_binary = self._prepare_bev_binary(bev_binary > 0)

        bev_rows = self._collect_row_samples(bev_binary)
        if len(bev_rows) < self.cfg.min_points:
            return self._empty_result(
                message="BEV 中有效边界点不足，无法生成稳定导航线。",
                image_hw=(image_height, image_width),
                foreground_ratio=foreground_ratio,
                image_rows=image_rows,
                source_quad=source_quad,
                destination_quad=destination_quad,
            )

        fit = self._fit_bev_centerline(bev_rows, bev_width, bev_height)
        if fit is None:
            return self._empty_result(
                message="BEV 中心线拟合失败。",
                image_hw=(image_height, image_width),
                foreground_ratio=foreground_ratio,
                image_rows=image_rows,
                source_quad=source_quad,
                destination_quad=destination_quad,
            )

        projected_centerline = self._project_points(fit["centerline_bev"], inverse_matrix)
        projected_left = self._project_points(fit["left_boundary_bev"], inverse_matrix)
        projected_right = self._project_points(fit["right_boundary_bev"], inverse_matrix)
        projected_bottom = self._project_points([fit["bottom_point_bev"]], inverse_matrix)
        projected_lookahead = self._project_points([fit["lookahead_point_bev"]], inverse_matrix)

        source_width_top = float(source_quad[1][0] - source_quad[0][0])
        source_width_bottom = float(source_quad[2][0] - source_quad[3][0])
        perspective_score = max(0.0, min(1.0, source_width_top / max(source_width_bottom, 1.0)))
        confidence = self._estimate_confidence(
            row_count=len(fit["inlier_rows"]),
            bev_height=bev_height,
            corridor_width_px=fit["corridor_width_px"],
            fit_rmse=fit["fit_rmse"],
            perspective_score=perspective_score,
        )

        return {
            "status": "ok",
            "message": "BEV 导航线拟合完成。",
            "image_size": {"width": image_width, "height": image_height},
            "bev_size": {"width": bev_width, "height": bev_height},
            "foreground_ratio": round(foreground_ratio, 6),
            "point_count": len(fit["inlier_rows"]),
            "fit_rmse": round(fit["fit_rmse"], 4),
            "confidence": round(confidence, 4),
            "corridor_width_px": round(fit["corridor_width_px"], 3),
            "lateral_error_px": round(fit["lateral_error_px"], 3),
            "center_offset_px": round(fit["lateral_error_px"], 3),
            "lookahead_error_px": round(fit["lookahead_error_px"], 3),
            "heading_angle_deg": round(fit["heading_angle_deg"], 3),
            "bottom_point_bev": self._point_to_list(fit["bottom_point_bev"]),
            "lookahead_point_bev": self._point_to_list(fit["lookahead_point_bev"]),
            "bottom_point": self._point_to_list(projected_bottom[0]) if projected_bottom else None,
            "lookahead_point": self._point_to_list(projected_lookahead[0]) if projected_lookahead else None,
            "source_quad": [self._point_to_list(point) for point in source_quad.tolist()],
            "destination_quad": [self._point_to_list(point) for point in destination_quad.tolist()],
            "centerline_bev": [self._point_to_list(point) for point in fit["centerline_bev"]],
            "projected_centerline": [self._point_to_list(point) for point in projected_centerline],
            "left_boundary_bev": [self._point_to_list(point) for point in fit["left_boundary_bev"]],
            "right_boundary_bev": [self._point_to_list(point) for point in fit["right_boundary_bev"]],
            "projected_left_boundary": [self._point_to_list(point) for point in projected_left],
            "projected_right_boundary": [self._point_to_list(point) for point in projected_right],
            "sample_points": [
                self._point_to_list((row["center"], row["y"])) for row in fit["inlier_rows"]
            ],
            "raw_sample_points": [
                self._point_to_list((row["center"], row["y"])) for row in bev_rows
            ],
        }

    def _prepare_binary(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        roi_top = int(np.clip(round(binary.shape[0] * self.cfg.roi_top_ratio), 0, binary.shape[0] - 1))
        if roi_top > 0:
            binary[:roi_top, :] = 0

        kernel_size = max(1, int(self.cfg.morphology_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if self.cfg.keep_largest_component:
            binary = self._keep_largest_component(binary)
        return binary

    def _prepare_bev_binary(self, binary: np.ndarray) -> np.ndarray:
        bev_binary = binary.astype(np.uint8)
        kernel_size = max(3, int(self.cfg.morphology_kernel_size // 2) | 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        bev_binary = cv2.morphologyEx(bev_binary, cv2.MORPH_CLOSE, kernel)
        if self.cfg.keep_largest_component:
            bev_binary = self._keep_largest_component(bev_binary)
        return bev_binary

    @staticmethod
    def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        if count <= 1:
            return binary.astype(np.uint8)

        largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest_index).astype(np.uint8)

    def _collect_row_samples(self, binary: np.ndarray) -> list[dict[str, float]]:
        height, width = binary.shape
        step = max(1, int(self.cfg.scan_step))
        min_width = max(6, int(round(width * self.cfg.min_segment_width_ratio)))

        rows: list[dict[str, float]] = []
        previous_center: float | None = None
        previous_width: float | None = None

        for y in range(height - 1, -1, -step):
            indices = np.flatnonzero(binary[y] > 0)
            if indices.size == 0:
                continue

            segments = self._segments_from_indices(indices)
            segment = self._pick_best_segment(
                segments=segments,
                min_width=min_width,
                previous_center=previous_center,
                previous_width=previous_width,
                image_width=width,
            )
            if segment is None:
                continue

            left, right = segment
            corridor_width = float(right - left + 1)
            center = float((left + right) / 2.0)

            if previous_center is not None:
                max_center_jump = max(width * float(self.cfg.center_jump_ratio), corridor_width * 0.55)
                if abs(center - previous_center) > max_center_jump:
                    continue
            if previous_width is not None:
                max_width_jump = max(width * 0.05, previous_width * float(self.cfg.width_jump_ratio))
                if abs(corridor_width - previous_width) > max_width_jump:
                    continue

            previous_center = center
            previous_width = corridor_width
            rows.append(
                {
                    "y": float(y),
                    "left": float(left),
                    "right": float(right),
                    "center": center,
                    "width": corridor_width,
                }
            )

        return rows

    @staticmethod
    def _segments_from_indices(indices: np.ndarray) -> list[tuple[int, int]]:
        if indices.size == 0:
            return []

        breaks = np.where(np.diff(indices) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, indices.size - 1]
        return [(int(indices[start]), int(indices[end])) for start, end in zip(starts, ends)]

    def _pick_best_segment(
        self,
        segments: list[tuple[int, int]],
        min_width: int,
        previous_center: float | None,
        previous_width: float | None,
        image_width: int,
    ) -> tuple[int, int] | None:
        candidates = [segment for segment in segments if segment[1] - segment[0] + 1 >= min_width]
        if not candidates:
            return None

        reference_center = previous_center if previous_center is not None else image_width / 2.0

        def score(segment: tuple[int, int]) -> float:
            left, right = segment
            width = right - left + 1
            center = (left + right) / 2.0
            score_value = width * 2.2 - abs(center - reference_center) * 1.1
            if previous_width is not None:
                score_value -= abs(width - previous_width) * 0.6
            return score_value

        return max(candidates, key=score)

    def _estimate_source_quad(
        self,
        image_rows: list[dict[str, float]],
        image_width: int,
    ) -> np.ndarray | None:
        if len(image_rows) < self.cfg.min_points:
            return None

        rows_desc = sorted(image_rows, key=lambda item: item["y"], reverse=True)
        top_band = max(3, int(len(rows_desc) * float(self.cfg.source_top_band_ratio)))
        bottom_band = max(3, int(len(rows_desc) * float(self.cfg.source_bottom_band_ratio)))

        bottom_rows = rows_desc[:bottom_band]
        top_rows = rows_desc[-top_band:]

        left_top = float(np.median([row["left"] for row in top_rows]))
        right_top = float(np.median([row["right"] for row in top_rows]))
        y_top = float(np.median([row["y"] for row in top_rows]))

        left_bottom = float(np.median([row["left"] for row in bottom_rows]))
        right_bottom = float(np.median([row["right"] for row in bottom_rows]))
        y_bottom = float(np.median([row["y"] for row in bottom_rows]))

        min_corridor_width = image_width * float(self.cfg.min_corridor_width_ratio)
        if (right_bottom - left_bottom) < min_corridor_width or (right_top - left_top) < min_corridor_width * 0.45:
            return None

        if y_bottom <= y_top:
            return None

        quad = np.array(
            [
                [left_top, y_top],
                [right_top, y_top],
                [right_bottom, y_bottom],
                [left_bottom, y_bottom],
            ],
            dtype=np.float32,
        )
        quad[:, 0] = np.clip(quad[:, 0], 0, image_width - 1)
        return quad

    def _destination_quad(self, width: int, height: int) -> np.ndarray:
        margin = width * float(self.cfg.bev_margin_ratio)
        return np.array(
            [
                [margin, 0],
                [width - 1 - margin, 0],
                [width - 1 - margin, height - 1],
                [margin, height - 1],
            ],
            dtype=np.float32,
        )

    def _fit_bev_centerline(
        self,
        bev_rows: list[dict[str, float]],
        bev_width: int,
        bev_height: int,
    ) -> dict[str, Any] | None:
        if len(bev_rows) < self.cfg.min_points:
            return None

        ys = np.array([row["y"] for row in bev_rows], dtype=np.float64)
        xs = np.array([row["center"] for row in bev_rows], dtype=np.float64)
        lefts = np.array([row["left"] for row in bev_rows], dtype=np.float64)
        rights = np.array([row["right"] for row in bev_rows], dtype=np.float64)
        widths = np.array([row["width"] for row in bev_rows], dtype=np.float64)

        degree = max(1, min(int(self.cfg.polynomial_degree), len(bev_rows) - 1))
        weights = widths * np.linspace(1.4, 1.0, num=len(widths))
        inlier_mask = np.ones(len(bev_rows), dtype=bool)

        for _ in range(3):
            if int(inlier_mask.sum()) < self.cfg.min_points:
                break

            coefficients = np.polyfit(ys[inlier_mask], xs[inlier_mask], deg=degree, w=weights[inlier_mask])
            residuals = np.abs(xs - np.polyval(coefficients, ys))
            threshold = max(float(self.cfg.fit_outlier_threshold_px), float(np.median(widths) * 0.16))
            refined_mask = residuals <= threshold
            if refined_mask.sum() == inlier_mask.sum():
                inlier_mask = refined_mask
                break
            inlier_mask = refined_mask

        if int(inlier_mask.sum()) < self.cfg.min_points:
            inlier_mask[:] = True

        coefficients = np.polyfit(ys[inlier_mask], xs[inlier_mask], deg=degree, w=weights[inlier_mask])
        fit_xs = np.polyval(coefficients, ys[inlier_mask])
        fit_rmse = float(np.sqrt(np.mean(np.square(xs[inlier_mask] - fit_xs))))

        y_top = float(np.min(ys[inlier_mask]))
        y_bottom = float(bev_height - 1)
        eval_ys = np.linspace(y_bottom, y_top, num=max(2, int(self.cfg.path_point_count)))
        eval_xs = np.clip(np.polyval(coefficients, eval_ys), 0, bev_width - 1)

        bottom_x = float(np.clip(np.polyval(coefficients, y_bottom), 0, bev_width - 1))
        lookahead_y = float(np.clip(round(bev_height * self.cfg.lookahead_ratio), y_top, y_bottom - 1))
        lookahead_x = float(np.clip(np.polyval(coefficients, lookahead_y), 0, bev_width - 1))

        dy = y_bottom - lookahead_y
        heading_angle_deg = float(np.degrees(np.arctan2(lookahead_x - bottom_x, dy))) if dy > 0 else 0.0
        lateral_error_px = float(bottom_x - (bev_width - 1) / 2.0)
        lookahead_error_px = float(lookahead_x - (bev_width - 1) / 2.0)
        corridor_width_px = float(np.median(widths[inlier_mask]))

        inlier_rows = [row for row, keep in zip(bev_rows, inlier_mask.tolist()) if keep]
        left_boundary = [(row["left"], row["y"]) for row in inlier_rows]
        right_boundary = [(row["right"], row["y"]) for row in inlier_rows]

        return {
            "fit_rmse": fit_rmse,
            "corridor_width_px": corridor_width_px,
            "lateral_error_px": lateral_error_px,
            "lookahead_error_px": lookahead_error_px,
            "heading_angle_deg": heading_angle_deg,
            "centerline_bev": [(float(x), float(y)) for x, y in zip(eval_xs.tolist(), eval_ys.tolist())],
            "left_boundary_bev": left_boundary,
            "right_boundary_bev": right_boundary,
            "bottom_point_bev": (bottom_x, y_bottom),
            "lookahead_point_bev": (lookahead_x, lookahead_y),
            "inlier_rows": inlier_rows,
        }

    def _estimate_confidence(
        self,
        row_count: int,
        bev_height: int,
        corridor_width_px: float,
        fit_rmse: float,
        perspective_score: float,
    ) -> float:
        expected_rows = max(1, bev_height // max(1, int(self.cfg.scan_step)))
        coverage_score = min(1.0, row_count / expected_rows)
        width_score = min(1.0, corridor_width_px / max(1.0, self.cfg.bev_width * 0.24))
        fit_score = max(0.0, 1.0 - fit_rmse / max(1.0, self.cfg.bev_width * self.cfg.max_fit_error_ratio))
        return max(0.0, min(1.0, coverage_score * 0.35 + width_score * 0.25 + fit_score * 0.25 + perspective_score * 0.15))

    def _build_navigation_overlays(
        self,
        image_path: Path,
        overlay_path: Path | None,
        mask: np.ndarray,
        result: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        base_image = None
        if overlay_path is not None and overlay_path.exists():
            base_image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if base_image is None:
            base_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if base_image is None:
            base_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if base_image.shape[:2] != mask.shape:
            base_image = cv2.resize(base_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        nav_overlay = base_image.copy()
        bev_overlay = np.full((int(self.cfg.bev_height), int(self.cfg.bev_width), 3), 244, dtype=np.uint8)

        if result.get("status") == "ok":
            source_quad = np.array(result.get("source_quad") or [], dtype=np.float32)
            if source_quad.size == 8:
                quad = source_quad.reshape(-1, 2).astype(np.int32)
                cv2.polylines(nav_overlay, [quad], isClosed=True, color=(240, 190, 0), thickness=2, lineType=cv2.LINE_AA)

            projected_centerline = result.get("projected_centerline") or []
            self._draw_polyline(nav_overlay, projected_centerline, (40, 60, 235), 4)
            self._draw_polyline(nav_overlay, result.get("projected_left_boundary") or [], (210, 120, 40), 2)
            self._draw_polyline(nav_overlay, result.get("projected_right_boundary") or [], (210, 120, 40), 2)
            self._draw_points(nav_overlay, result.get("sample_points") or [], (255, 180, 0), radius=3)
            self._draw_highlight_point(nav_overlay, result.get("bottom_point"), (0, 255, 0))
            self._draw_highlight_point(nav_overlay, result.get("lookahead_point"), (0, 140, 255))

            center_x = int(round((mask.shape[1] - 1) / 2.0))
            cv2.line(nav_overlay, (center_x, 0), (center_x, mask.shape[0] - 1), (0, 255, 255), 1, cv2.LINE_AA)

            if source_quad.size == 8:
                destination_quad = np.array(result.get("destination_quad") or [], dtype=np.float32)
                if destination_quad.size == 8:
                    homography = cv2.getPerspectiveTransform(source_quad.reshape(4, 2), destination_quad.reshape(4, 2))
                    warped = cv2.warpPerspective(base_image, homography, (int(self.cfg.bev_width), int(self.cfg.bev_height)))
                    bev_overlay = warped

            self._draw_polyline(bev_overlay, result.get("centerline_bev") or [], (40, 60, 235), 4)
            self._draw_polyline(bev_overlay, result.get("left_boundary_bev") or [], (210, 120, 40), 2)
            self._draw_polyline(bev_overlay, result.get("right_boundary_bev") or [], (210, 120, 40), 2)
            self._draw_points(bev_overlay, result.get("raw_sample_points") or [], (255, 180, 0), radius=2)
            self._draw_highlight_point(bev_overlay, result.get("bottom_point_bev"), (0, 255, 0))
            self._draw_highlight_point(bev_overlay, result.get("lookahead_point_bev"), (0, 140, 255))

            bev_center_x = int(round((self.cfg.bev_width - 1) / 2.0))
            cv2.line(bev_overlay, (bev_center_x, 0), (bev_center_x, int(self.cfg.bev_height) - 1), (0, 255, 255), 1, cv2.LINE_AA)

            text_lines = [
                f"lateral_px: {result.get('lateral_error_px', 0):.1f}",
                f"heading_deg: {result.get('heading_angle_deg', 0):.1f}",
                f"width_px: {result.get('corridor_width_px', 0):.1f}",
                f"confidence: {result.get('confidence', 0):.2f}",
            ]
        else:
            text_lines = [str(result.get("message") or "navigation failed")]

        self._draw_text_block(nav_overlay, text_lines)
        self._draw_text_block(bev_overlay, text_lines)
        return nav_overlay, bev_overlay

    def _resolve_output_paths(self, image_path: Path, suffix: str) -> tuple[Path, Path, Path]:
        output_dir = self.cfg.output_dir
        if output_dir is None:
            dataset_paths = find_dataset_paths_for_image(self._project_cfg, image_path)
            if dataset_paths is not None:
                output_dir = dataset_paths.navigation_dir
            else:
                output_dir = image_path.parent / self._project_cfg.dataset.navigation_dirname

        result_stem = f"{image_path.stem}_{suffix}" if suffix else image_path.stem
        nav_overlay_path = output_dir / f"{result_stem}{self.cfg.overlay_suffix}"
        bev_overlay_path = output_dir / f"{result_stem}{self.cfg.bev_overlay_suffix}"
        nav_json_path = output_dir / f"{result_stem}{self.cfg.json_suffix}"
        return nav_overlay_path, bev_overlay_path, nav_json_path

    @staticmethod
    def _project_points(points: list[tuple[float, float]] | list[list[float]] | list[Any], matrix: np.ndarray) -> list[tuple[float, float]]:
        if not points:
            return []
        array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(array, matrix).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in projected.tolist()]

    @staticmethod
    def _point_to_list(point: tuple[float, float] | list[float] | np.ndarray) -> list[float]:
        return [round(float(point[0]), 2), round(float(point[1]), 2)]

    def _empty_result(
        self,
        message: str,
        image_hw: tuple[int, int],
        foreground_ratio: float,
        image_rows: list[dict[str, float]],
        source_quad: np.ndarray | None = None,
        destination_quad: np.ndarray | None = None,
    ) -> dict[str, Any]:
        image_height, image_width = image_hw
        return {
            "status": "failed",
            "message": message,
            "image_size": {"width": image_width, "height": image_height},
            "bev_size": {"width": int(self.cfg.bev_width), "height": int(self.cfg.bev_height)},
            "foreground_ratio": round(foreground_ratio, 6),
            "point_count": len(image_rows),
            "fit_rmse": None,
            "confidence": 0.0,
            "corridor_width_px": None,
            "lateral_error_px": None,
            "center_offset_px": None,
            "lookahead_error_px": None,
            "heading_angle_deg": None,
            "bottom_point_bev": None,
            "lookahead_point_bev": None,
            "bottom_point": None,
            "lookahead_point": None,
            "source_quad": [self._point_to_list(point) for point in source_quad.tolist()] if source_quad is not None else [],
            "destination_quad": [self._point_to_list(point) for point in destination_quad.tolist()] if destination_quad is not None else [],
            "centerline_bev": [],
            "projected_centerline": [],
            "left_boundary_bev": [],
            "right_boundary_bev": [],
            "projected_left_boundary": [],
            "projected_right_boundary": [],
            "sample_points": [],
            "raw_sample_points": [self._point_to_list((row["center"], row["y"])) for row in image_rows],
        }

    @staticmethod
    def _draw_polyline(
        canvas: np.ndarray,
        points: list[list[float]] | list[tuple[float, float]],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if len(points) < 2:
            return
        poly = np.array([[int(round(point[0])), int(round(point[1]))] for point in points], dtype=np.int32)
        cv2.polylines(canvas, [poly], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_points(
        canvas: np.ndarray,
        points: list[list[float]] | list[tuple[float, float]],
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        for point in points:
            cv2.circle(
                canvas,
                (int(round(point[0])), int(round(point[1]))),
                radius,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    @staticmethod
    def _draw_highlight_point(
        canvas: np.ndarray,
        point: list[float] | tuple[float, float] | None,
        color: tuple[int, int, int],
    ) -> None:
        if point is None:
            return
        cv2.circle(
            canvas,
            (int(round(point[0])), int(round(point[1]))),
            7,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _draw_text_block(canvas: np.ndarray, lines: list[str]) -> None:
        base_y = 34
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (18, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (24, 28, 32),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (18, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (248, 248, 245),
                1,
                cv2.LINE_AA,
            )
            base_y += 30
