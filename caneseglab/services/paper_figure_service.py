from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..artifacts import latest_run_dir
from ..config import PaperFigureTemplateConfig, ProjectConfig
from .log_service import logger


@dataclass(frozen=True)
class NavigationRecord:
    """导航结果统计记录。"""

    dataset_name: str
    status: str
    confidence: float | None
    fit_rmse: float | None
    lateral_error_px: float | None
    heading_angle_deg: float | None

    @classmethod
    def from_file(cls, path: Path) -> "NavigationRecord":
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        return cls(
            dataset_name=path.parent.parent.name,
            status=str(result.get("status", "unknown")),
            confidence=_to_optional_float(result.get("confidence")),
            fit_rmse=_to_optional_float(result.get("fit_rmse")),
            lateral_error_px=_to_optional_float(result.get("lateral_error_px")),
            heading_angle_deg=_to_optional_float(result.get("heading_angle_deg")),
        )


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class PaperFigureService:
    """论文图导出服务。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.project_cfg = config
        self.cfg = config.paper_figures

    def export(
        self,
        history_path: Path | None = None,
        navigation_root: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        plt = self._prepare_matplotlib()
        export_dir = (output_dir or self.cfg.output_dir).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(export_dir),
            "history_path": None,
            "navigation_root": None,
            "figures": {},
        }

        resolved_history = self._resolve_history_path(history_path)
        if resolved_history is not None and resolved_history.exists():
            history_payload = self._load_history_payload(resolved_history)
            manifest["history_path"] = str(resolved_history.resolve())
            manifest["figures"]["training_loss"] = self._plot_training_loss(
                plt=plt,
                output_dir=export_dir,
                history_payload=history_payload,
            )
            manifest["figures"]["validation_metrics"] = self._plot_validation_metrics(
                plt=plt,
                output_dir=export_dir,
                history_payload=history_payload,
            )
            manifest["figures"]["learning_rate"] = self._plot_learning_rate(
                plt=plt,
                output_dir=export_dir,
                history_payload=history_payload,
            )

        resolved_navigation_root = (navigation_root or self.project_cfg.dataset.root_dir).resolve()
        if resolved_navigation_root.exists():
            nav_paths = sorted(
                resolved_navigation_root.glob("*/navigation/*_nav.json"),
                key=lambda item: str(item).lower(),
            )
            manifest["navigation_root"] = str(resolved_navigation_root)
            if nav_paths:
                records = [NavigationRecord.from_file(path) for path in nav_paths]
                manifest["figures"]["navigation_status"] = self._plot_navigation_status(
                    plt=plt,
                    output_dir=export_dir,
                    records=records,
                )
                manifest["figures"]["navigation_metrics"] = self._plot_navigation_metrics(
                    plt=plt,
                    output_dir=export_dir,
                    records=records,
                )

        manifest_path = export_dir / "figure_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("论文图导出完成，清单已保存：%s", manifest_path)
        return manifest

    def _resolve_history_path(self, history_path: Path | None) -> Path | None:
        if history_path is not None:
            return history_path

        try:
            return latest_run_dir(self.project_cfg.train.output_dir) / "history.json"
        except FileNotFoundError:
            return None

    @staticmethod
    def _load_history_payload(path: Path) -> dict[str, Any]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        epochs = raw.get("epochs")
        if isinstance(epochs, list) and epochs:
            train_loss = [float(item.get("train_loss", 0.0)) for item in epochs]
            valid_loss = [float(item.get("valid_loss", 0.0)) for item in epochs]
            valid_dice = [float(item.get("valid_dice", 0.0)) for item in epochs]
            valid_iou = [_to_optional_float(item.get("valid_iou")) for item in epochs]
            learning_rate = [_to_optional_float(item.get("lr")) for item in epochs]
        else:
            train_loss = [float(item) for item in raw.get("train_loss", [])]
            valid_loss = [float(item) for item in raw.get("valid_loss", [])]
            valid_dice = [float(item) for item in raw.get("valid_dice", [])]
            valid_iou = [_to_optional_float(item) for item in raw.get("valid_iou", [])]
            learning_rate = [_to_optional_float(item) for item in raw.get("learning_rate", [])]

        best_epoch = _to_optional_float(raw.get("artifacts", {}).get("best_epoch"))
        if best_epoch is None and valid_dice:
            best_epoch = int(max(range(len(valid_dice)), key=lambda index: valid_dice[index])) + 1

        return {
            "meta": raw.get("meta", {}),
            "dataset": raw.get("dataset", {}),
            "training": raw.get("training", {}),
            "artifacts": raw.get("artifacts", {}),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_dice": valid_dice,
            "valid_iou": valid_iou,
            "learning_rate": learning_rate,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
        }

    def _plot_training_loss(
        self,
        plt,
        output_dir: Path,
        history_payload: dict[str, Any],
    ) -> list[str]:
        template = self._uniform_training_template()
        figure, axis = plt.subplots(figsize=(template.width_in, template.height_in))
        train_epochs = list(range(1, len(history_payload["train_loss"]) + 1))
        valid_loss_epochs = list(range(1, len(history_payload["valid_loss"]) + 1))
        colors = list(self.cfg.color_cycle)
        line_width = template.line_width or self.cfg.line_width

        axis.plot(
            train_epochs,
            history_payload["train_loss"],
            label="训练损失",
            linewidth=line_width,
            color=colors[0],
            marker="o",
            markersize=self.cfg.marker_size * 0.65,
        )
        axis.plot(
            valid_loss_epochs,
            history_payload["valid_loss"],
            label="验证损失",
            linewidth=line_width,
            color=colors[1],
            marker="s",
            markersize=self.cfg.marker_size * 0.65,
        )
        axis.set_xlabel("Epoch", fontsize=self.cfg.label_size)
        axis.set_ylabel("Loss", fontsize=self.cfg.label_size)
        self._style_axis(axis, template)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc=template.legend_loc, fontsize=self.cfg.legend_size, frameon=False)
        if self.cfg.tight_layout:
            figure.tight_layout()

        return self._save_figure(plt, figure, output_dir, "fig01_training_loss")

    def _plot_validation_metrics(
        self,
        plt,
        output_dir: Path,
        history_payload: dict[str, Any],
    ) -> list[str]:
        template = self._uniform_training_template()
        figure, axis = plt.subplots(figsize=(template.width_in, template.height_in))
        colors = list(self.cfg.color_cycle)
        line_width = template.line_width or self.cfg.line_width

        valid_dice_epochs = list(range(1, len(history_payload["valid_dice"]) + 1))
        if history_payload["valid_dice"]:
            axis.plot(
                valid_dice_epochs,
                history_payload["valid_dice"],
                label="验证 Dice",
                linewidth=line_width,
                color=colors[2],
                marker="o",
                markersize=self.cfg.marker_size * 0.65,
            )

        valid_iou_points = [
            (index + 1, value)
            for index, value in enumerate(history_payload["valid_iou"])
            if value is not None
        ]
        if valid_iou_points:
            axis.plot(
                [epoch for epoch, _ in valid_iou_points],
                [value for _, value in valid_iou_points],
                label="验证 IoU",
                linewidth=line_width,
                color=colors[3],
                marker="^",
                markersize=self.cfg.marker_size * 0.65,
            )

        best_epoch = history_payload.get("best_epoch")
        if best_epoch is not None and 1 <= best_epoch <= len(history_payload["valid_dice"]):
            best_dice = history_payload["valid_dice"][best_epoch - 1]
            axis.scatter(
                [best_epoch],
                [best_dice],
                color=colors[1],
                s=36,
                zorder=4,
                label=f"最佳 Epoch={best_epoch}",
            )

        if not valid_dice_epochs and not valid_iou_points:
            self._annotate_empty_axis(axis, "暂无验证指标记录")

        axis.set_xlabel("Epoch", fontsize=self.cfg.label_size)
        axis.set_ylabel("Score", fontsize=self.cfg.label_size)
        self._style_axis(axis, template)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc=template.legend_loc, fontsize=self.cfg.legend_size, frameon=False)
        if self.cfg.tight_layout:
            figure.tight_layout()

        return self._save_figure(plt, figure, output_dir, "fig02_validation_metrics")

    def _plot_learning_rate(
        self,
        plt,
        output_dir: Path,
        history_payload: dict[str, Any],
    ) -> list[str]:
        template = self._uniform_training_template()
        figure, axis = plt.subplots(figsize=(template.width_in, template.height_in))
        colors = list(self.cfg.color_cycle)
        line_width = template.line_width or self.cfg.line_width

        learning_rate_points = [
            (index + 1, value)
            for index, value in enumerate(history_payload["learning_rate"])
            if value is not None
        ]
        if learning_rate_points:
            axis.plot(
                [epoch for epoch, _ in learning_rate_points],
                [value for _, value in learning_rate_points],
                label="学习率",
                linewidth=line_width,
                color=colors[4],
                marker="d",
                markersize=self.cfg.marker_size * 0.65,
            )
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        else:
            self._annotate_empty_axis(axis, "暂无学习率记录")

        axis.set_xlabel("Epoch", fontsize=self.cfg.label_size)
        axis.set_ylabel("LR", fontsize=self.cfg.label_size)
        self._style_axis(axis, template)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc=template.legend_loc, fontsize=self.cfg.legend_size, frameon=False)
        if self.cfg.tight_layout:
            figure.tight_layout()

        return self._save_figure(plt, figure, output_dir, "fig03_learning_rate")

    def _plot_navigation_status(
        self,
        plt,
        output_dir: Path,
        records: list[NavigationRecord],
    ) -> list[str]:
        template = self._template("status_bar")
        figure, axis = plt.subplots(figsize=(template.width_in, template.height_in))

        dataset_order = sorted({record.dataset_name for record in records})
        ok_counts = []
        failed_counts = []
        for dataset_name in dataset_order:
            ok_counts.append(sum(1 for record in records if record.dataset_name == dataset_name and record.status == "ok"))
            failed_counts.append(sum(1 for record in records if record.dataset_name == dataset_name and record.status != "ok"))

        colors = list(self.cfg.color_cycle)
        axis.bar(dataset_order, ok_counts, color=colors[2], label="成功")
        axis.bar(dataset_order, failed_counts, bottom=ok_counts, color=colors[1], label="失败/异常")

        axis.set_title("各数据集导航结果统计", fontsize=self.cfg.title_size)
        axis.set_xlabel("数据集", fontsize=self.cfg.label_size)
        axis.set_ylabel("结果数量", fontsize=self.cfg.label_size)
        self._style_axis(axis, template)
        axis.legend(loc=template.legend_loc, fontsize=self.cfg.legend_size, frameon=False)

        if self.cfg.tight_layout:
            figure.tight_layout()

        return self._save_figure(plt, figure, output_dir, "fig04_navigation_status")

    def _plot_navigation_metrics(
        self,
        plt,
        output_dir: Path,
        records: list[NavigationRecord],
    ) -> list[str]:
        template = self._template("metric_grid")
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(template.width_in, template.height_in),
        )

        metric_map = {
            "置信度": [record.confidence for record in records if record.confidence is not None],
            "拟合误差 RMSE (px)": [record.fit_rmse for record in records if record.fit_rmse is not None],
            "横向误差 (px)": [record.lateral_error_px for record in records if record.lateral_error_px is not None],
            "航向角 (deg)": [record.heading_angle_deg for record in records if record.heading_angle_deg is not None],
        }
        colors = list(self.cfg.color_cycle)

        for index, (title, values) in enumerate(metric_map.items()):
            axis = axes.flat[index]
            axis.hist(
                values,
                bins=template.bins,
                color=colors[index % len(colors)],
                alpha=0.85,
                edgecolor="#FFFFFF",
                linewidth=0.6,
            )
            axis.set_title(title, fontsize=self.cfg.title_size)
            axis.set_xlabel(title, fontsize=self.cfg.label_size)
            axis.set_ylabel("频数", fontsize=self.cfg.label_size)
            self._style_axis(axis, template)

        if self.cfg.tight_layout:
            figure.tight_layout()

        return self._save_figure(plt, figure, output_dir, "fig05_navigation_metrics")

    def _style_axis(self, axis, template: PaperFigureTemplateConfig) -> None:
        axis.set_facecolor(self.cfg.axes_facecolor)
        axis.tick_params(
            axis="x",
            labelsize=self.cfg.tick_size,
            rotation=template.x_rotation,
        )
        axis.tick_params(
            axis="y",
            labelsize=self.cfg.tick_size,
            rotation=template.y_rotation,
        )
        if self.cfg.grid_enabled and template.show_grid:
            axis.grid(
                True,
                linestyle="--",
                alpha=self.cfg.grid_alpha,
                linewidth=0.7,
            )
        for spine in axis.spines.values():
            spine.set_linewidth(self.cfg.spine_width)

    @staticmethod
    def _annotate_empty_axis(axis, text: str) -> None:
        axis.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    def _save_figure(self, plt, figure, output_dir: Path, stem: str) -> list[str]:
        saved_paths: list[str] = []
        for fmt in self.cfg.formats:
            output_path = output_dir / f"{stem}.{fmt}"
            figure.savefig(
                output_path,
                dpi=self.cfg.dpi,
                transparent=self.cfg.transparent,
                bbox_inches=self.cfg.bbox_inches,
                pad_inches=self.cfg.pad_inches,
                facecolor="none" if self.cfg.transparent else self.cfg.save_facecolor,
            )
            saved_paths.append(str(output_path.resolve()))
        plt.close(figure)
        return saved_paths

    def _template(self, name: str) -> PaperFigureTemplateConfig:
        return self.cfg.templates.get(name, PaperFigureTemplateConfig())

    def _uniform_training_template(self) -> PaperFigureTemplateConfig:
        return PaperFigureTemplateConfig(
            width_in=self.cfg.default_width_in,
            height_in=self.cfg.default_height_in,
            line_width=self.cfg.line_width,
            legend_loc="best",
            show_grid=True,
        )

    def _prepare_matplotlib(self):
        try:
            import matplotlib
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "导出论文图需要 matplotlib，请先执行 `python -m pip install -r requirements.txt`。"
            ) from exc

        matplotlib.use("Agg")
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = list(self.cfg.font_family_cn)
        matplotlib.rcParams["axes.unicode_minus"] = not self.cfg.minus_sign_fix
        matplotlib.rcParams["font.size"] = self.cfg.font_size
        matplotlib.rcParams["axes.titlesize"] = self.cfg.title_size
        matplotlib.rcParams["axes.labelsize"] = self.cfg.label_size
        matplotlib.rcParams["xtick.labelsize"] = self.cfg.tick_size
        matplotlib.rcParams["ytick.labelsize"] = self.cfg.tick_size
        matplotlib.rcParams["legend.fontsize"] = self.cfg.legend_size
        matplotlib.rcParams["axes.facecolor"] = self.cfg.axes_facecolor
        matplotlib.rcParams["figure.facecolor"] = self.cfg.background_color
        matplotlib.rcParams["savefig.facecolor"] = self.cfg.save_facecolor
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=list(self.cfg.color_cycle))

        import matplotlib.pyplot as plt

        return plt
