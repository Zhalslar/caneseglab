from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import ProjectConfig
from .services.auto_label_service import AutoLabelService
from .services.benchmark_service import BenchmarkService
from .services.dataset_inference_service import DatasetInferenceService
from .services.dataset_navigation_service import DatasetNavigationService
from .services.mask_service import LabelmeMaskGenerator
from .services.navigation_service import NavigationService
from .services.navigation_audit_service import NavigationAuditService
from .services.paper_figure_service import PaperFigureService
from .services.training_service import TrainingService
from .services.onnx_inference_service import OnnxInferenceService

if TYPE_CHECKING:
    from .services.tensorrt_inference_service import TensorRTInferenceService


class SegmentationProject:
    """
    项目主类。

    设计目标：
    - 统一完成所有服务类的实例化
    - 统一暴露面向外部调用的接口
    - 入口文件只与该主类交互
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.cfg = config
        self.mask_service = LabelmeMaskGenerator(self.cfg)
        self._auto_label_service: AutoLabelService | None = None
        self._benchmark_service: BenchmarkService | None = None
        self._dataset_inference_service: DatasetInferenceService | None = None
        self._dataset_navigation_service: DatasetNavigationService | None = None
        self._navigation_service: NavigationService | None = None
        self._navigation_audit_service: NavigationAuditService | None = None
        self._paper_figure_service: PaperFigureService | None = None
        self._training_service: TrainingService | None = None
        self._onnx_inference_service: OnnxInferenceService | None = None
        self._trt_inference_service: TensorRTInferenceService | None = None

    @property
    def auto_label_service(self) -> AutoLabelService:
        if self._auto_label_service is None:
            self._auto_label_service = AutoLabelService(self.cfg)
        return self._auto_label_service

    @property
    def benchmark_service(self) -> BenchmarkService:
        if self._benchmark_service is None:
            self._benchmark_service = BenchmarkService(self.cfg)
        return self._benchmark_service

    @property
    def training_service(self) -> TrainingService:
        if self._training_service is None:
            self._training_service = TrainingService(self.cfg)
        return self._training_service

    @property
    def dataset_inference_service(self) -> DatasetInferenceService:
        if self._dataset_inference_service is None:
            self._dataset_inference_service = DatasetInferenceService(self.cfg)
        return self._dataset_inference_service

    @property
    def dataset_navigation_service(self) -> DatasetNavigationService:
        if self._dataset_navigation_service is None:
            self._dataset_navigation_service = DatasetNavigationService(self.cfg)
        return self._dataset_navigation_service

    @property
    def navigation_service(self) -> NavigationService:
        if self._navigation_service is None:
            self._navigation_service = NavigationService(self.cfg)
        return self._navigation_service

    @property
    def paper_figure_service(self) -> PaperFigureService:
        if self._paper_figure_service is None:
            self._paper_figure_service = PaperFigureService(self.cfg)
        return self._paper_figure_service

    @property
    def navigation_audit_service(self) -> NavigationAuditService:
        if self._navigation_audit_service is None:
            self._navigation_audit_service = NavigationAuditService(self.cfg)
        return self._navigation_audit_service

    @property
    def onnx_inference_service(self) -> OnnxInferenceService:
        if self._onnx_inference_service is None:
            self._onnx_inference_service = OnnxInferenceService(self.cfg)
        return self._onnx_inference_service

    @property
    def trt_inference_service(self) -> TensorRTInferenceService:
        if self._trt_inference_service is None:
            from .services.tensorrt_inference_service import TensorRTInferenceService

            self._trt_inference_service = TensorRTInferenceService(self.cfg)
        return self._trt_inference_service

    def generate_masks(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        progress_callback=None,
    ):
        """
        生成标注掩码
        参数：
        - `input_dir`：输入目录
        - `output_dir`：输出目录
        """
        return self.mask_service.generate(
            input_dir,
            output_dir,
            progress_callback=progress_callback,
        )

    def auto_label(
        self,
        image_dir: Path,
        annotation_dir: Path | None = None,
        overlay_dir: Path | None = None,
        progress_callback=None,
    ):
        """使用当前模型为未标注图片生成 Labelme JSON。"""
        return self.auto_label_service.label_unannotated(
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            overlay_dir=overlay_dir,
            progress_callback=progress_callback,
        )

    def infer_dataset_onnx(
        self,
        image_dir: Path,
        overlay_dir: Path | None = None,
        progress_callback=None,
    ):
        """为数据集中缺失覆盖图的图片补齐 ONNX 推理结果。"""
        return self.dataset_inference_service.infer_missing_overlays(
            image_dir=image_dir,
            overlay_dir=overlay_dir,
            progress_callback=progress_callback,
        )

    def benchmark_inference(
        self,
        image_dir: Path,
        output_dir: Path | None = None,
        backends: list[str] | None = None,
        warmup_runs: int | None = None,
        timed_runs: int | None = None,
        max_images: int | None = None,
    ) -> dict:
        """执行推理测速。"""
        return self.benchmark_service.benchmark(
            image_dir=image_dir,
            output_dir=output_dir,
            backends=backends,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            max_images=max_images,
        )

    def navigate_dataset_onnx(
        self,
        image_dir: Path,
        progress_callback=None,
    ):
        """为数据集批量生成 ONNX 导航线结果。"""
        return self.dataset_navigation_service.navigate_missing_results(
            image_dir=image_dir,
            progress_callback=progress_callback,
        )

    def train(self, image_dir: Path, mask_dir: Path):
        """
        训练分割模型。

        参数：
        - `image_dir`：训练图片目录
        - `mask_dir`：训练掩码目录
        """
        return self.training_service.train(image_dir, mask_dir)

    def export_paper_figures(
        self,
        history_path: Path | None = None,
        navigation_root: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict:
        """导出论文图。"""
        return self.paper_figure_service.export(
            history_path=history_path,
            navigation_root=navigation_root,
            output_dir=output_dir,
        )

    def audit_navigation_results(
        self,
        navigation_root: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict:
        """审计导航结果。"""
        return self.navigation_audit_service.audit(
            navigation_root=navigation_root,
            output_dir=output_dir,
        )

    def export_onnx(self, onnx_output_path: Path | None = None) -> Path:
        """
        导出 ONNX 模型。

        参数：
        - `onnx_output_path`：ONNX 输出路径
        """
        return self.training_service.export_onnx(onnx_output_path)

    def export_tensorrt(
        self,
        onnx_path: Path | None = None,
        engine_output_path: Path | None = None,
    ) -> Path:
        """从 ONNX 导出 TensorRT 引擎。"""
        return self.trt_inference_service.export_engine(
            onnx_path=onnx_path,
            engine_output_path=engine_output_path,
        )

    def verify_onnx(self, image_path: Path | None = None) -> tuple[Path, Path]:
        """
        使用 ONNXRuntime 进行推理验证并导出可视化结果。
        参数：
        - `image_path`：待推理图片路径
        返回：
        - `(mask_output_path, overlay_output_path)`
        """
        target_path = image_path or self.cfg.inference.image_path
        return self.onnx_inference_service.infer_file(target_path)

    def infer_tensorrt(self, image_path: Path | None = None) -> tuple[Path, Path]:
        """
        使用 TensorRT 引擎推理并导出可视化结果。
        参数：
        - `image_path`：待推理图片路径
        返回：
        - `(mask_output_path, overlay_output_path)`
        """
        target_path = image_path or self.cfg.inference.image_path
        return self.trt_inference_service.infer_file(target_path)

    def navigate_onnx(self, image_path: Path | None = None) -> dict:
        """
        使用 ONNX 推理并生成导航线结果。
        """
        target_path = image_path or self.cfg.inference.image_path
        mask_output, overlay_output = self.onnx_inference_service.infer_file(target_path)
        return self.navigation_service.build_from_files(
            image_path=target_path,
            mask_path=mask_output,
            overlay_path=overlay_output,
            suffix="onnx",
        )

    def navigate_tensorrt(self, image_path: Path | None = None) -> dict:
        """
        使用 TensorRT 推理并生成导航线结果。
        """
        target_path = image_path or self.cfg.inference.image_path
        mask_output, overlay_output = self.trt_inference_service.infer_file(target_path)
        return self.navigation_service.build_from_files(
            image_path=target_path,
            mask_path=mask_output,
            overlay_path=overlay_output,
            suffix="trt",
        )
