from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_paper_figure_templates() -> dict[str, "PaperFigureTemplateConfig"]:
    return {
        "status_bar": PaperFigureTemplateConfig(
            width_in=6.2,
            height_in=4.0,
            legend_loc="best",
            show_grid=True,
        ),
        "metric_grid": PaperFigureTemplateConfig(
            width_in=8.0,
            height_in=5.8,
            show_grid=True,
            bins=20,
        ),
        "box_stat": PaperFigureTemplateConfig(
            width_in=6.2,
            height_in=4.0,
            show_grid=True,
        ),
        "image_panel_2x2": PaperFigureTemplateConfig(
            width_in=7.2,
            height_in=5.6,
            panel_spacing=0.12,
            caption_enabled=True,
        ),
        "bev_demo": PaperFigureTemplateConfig(
            width_in=7.0,
            height_in=3.8,
            panel_spacing=0.08,
            caption_enabled=True,
        ),
    }


@dataclass
class DatasetConfig:
    """数据集目录结构配置。"""

    """数据集根目录。"""
    root_dir: Path = Path("data/samples")
    """原图目录名。"""
    images_dirname: str = "images"
    """标注目录名。"""
    annotations_dirname: str = "annotations"
    """掩码目录名。"""
    masks_dirname: str = "masks"
    """推理覆盖图目录名。"""
    overlay_dirname: str = "overlays"
    """导航结果目录名。"""
    navigation_dirname: str = "navigation"


@dataclass
class MaskConfig:
    """标注转掩码配置。"""

    """掩码前景像素值。"""
    mask_value: int = 255
    """输出掩码文件名后缀。"""
    output_suffix: str = "_mask.png"
    """待处理标注文件匹配模式。"""
    json_pattern: str = "*.json"


@dataclass
class TrainConfig:
    """训练过程配置。"""

    """训练产物根目录。"""
    output_dir: Path = Path("data/train")
    """随机种子。"""
    random_seed: int = 101
    """模型名称。"""
    model_name: str = "Unet"
    """编码器骨干网络。"""
    backbone: str = "efficientnet-b0"
    """输入通道数。"""
    in_channels: int = 3
    """分割类别数。"""
    num_classes: int = 1
    """训练批大小。"""
    train_batch_size: int = 1
    """验证批大小。"""
    valid_batch_size: int = 4
    """数据加载线程数。"""
    num_workers: int = 8
    """训练轮数。"""
    epochs: int = 50
    """初始学习率。"""
    learning_rate: float = 2e-3
    """权重衰减系数。"""
    weight_decay: float = 1e-6
    """学习率下限。"""
    min_learning_rate: float = 1e-6
    """学习率调度器名称。"""
    scheduler_name: str = "cosine"
    """优化器名称。"""
    optimizer_name: str = "adam"
    """ReduceLROnPlateau 的耐心轮数。"""
    scheduler_patience: int = 3
    """梯度累积步数。"""
    grad_accum_steps: int = 1
    """是否启用自动混合精度。"""
    use_amp: bool = True
    """训练输入宽度。"""
    crop_width: int = 1024
    """训练输入高度。"""
    crop_height: int = 768
    """是否启用训练增强。"""
    enable_augmentation: bool = True
    """分割阈值。"""
    threshold: float = 0.5
    """训练设备，支持 auto/cpu/cuda/cuda:0。"""
    device: str = "cuda"
    """是否在训练结束后自动导出 ONNX。"""
    export_onnx: bool = False
    """BCE + Dice 中 Dice 项权重。"""
    dice_loss_weight: float = 1.0
    """训练指标计算阈值。"""
    metric_threshold: float = 0.5
    """水平翻转概率。"""
    horizontal_flip_prob: float = 0.25
    """亮度对比度增强概率。"""
    brightness_contrast_prob: float = 0.25
    """ONNX 导出 opset 版本。"""
    onnx_opset_version: int = 17
    """ONNX 是否启用动态 batch。"""
    onnx_dynamic_batch: bool = True
    """训练集划分比例。"""
    train_ratio: float = 0.8
    """验证集划分比例。"""
    valid_ratio: float = 0.1
    """支持的图像后缀。"""
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    """掩码文件名后缀。"""
    mask_suffix: str = "_mask.png"
    """二值分割阈值。"""
    binary_threshold: int = 127


@dataclass
class InferenceConfig:
    """推理与可视化配置"""

    """训练产物根目录。"""
    artifacts_dir: Path = Path("data/train")

    """ONNX 模型路径；为空时自动读取最近一次训练产物。"""
    onnx_path: Path | None = None
    """TensorRT 引擎路径；为空时自动读取最近一次训练产物。"""
    engine_path: Path | None = None

    """默认输入图像路径。"""
    image_path: Path = Path("data/samples/demo/images/IMG_20260110_102451.jpg")

    """掩码输出路径；为空时默认保存到数据集 masks 目录。"""
    mask_output: Path | None = None
    """叠加图输出路径；为空时默认保存到数据集 overlays 目录。"""
    overlay_output: Path | None = None

    """掩码阈值。"""
    threshold: float = 0.5
    """推理缩放宽度；为空时按模型输入决定。"""
    resize_width: int | None = 1024
    """推理缩放高度；为空时按模型输入决定。"""
    resize_height: int | None = 768

    """叠加显示颜色，格式为 BGR。"""
    overlay_color: tuple[int, int, int] = (0, 255, 0)  # BGR
    """叠加透明度。"""
    overlay_alpha: float = 0.35


@dataclass
class NavigationConfig:
    """导航线拟合配置。"""

    """导航结果输出目录；为空时跟随数据集 navigation 目录。"""
    output_dir: Path | None = None
    """原图扫描区域顶部比例。"""
    roi_top_ratio: float = 0.35
    """逐行扫描步长。"""
    scan_step: int = 12
    """最小前景带宽度比例。"""
    min_segment_width_ratio: float = 0.03
    """最少有效采样点数。"""
    min_points: int = 10
    """导航线拟合阶数；在 BEV 中通常使用 1 次更稳。"""
    polynomial_degree: int = 1
    """输出导航线采样点数。"""
    path_point_count: int = 24
    """航向角前视点比例；相对于 BEV 高度。"""
    lookahead_ratio: float = 0.58
    """形态学闭运算核尺寸。"""
    morphology_kernel_size: int = 9
    """最大拟合误差比例。"""
    max_fit_error_ratio: float = 0.08
    """是否仅保留最大连通域。"""
    keep_largest_component: bool = True
    """是否绘制采样点。"""
    draw_sample_points: bool = False
    """BEV 输出宽度。"""
    bev_width: int = 960
    """BEV 输出高度。"""
    bev_height: int = 1280
    """BEV 目标矩形左右边距比例。"""
    bev_margin_ratio: float = 0.18
    """源区域顶部锚点带占比。"""
    source_top_band_ratio: float = 0.2
    """源区域底部锚点带占比。"""
    source_bottom_band_ratio: float = 0.16
    """相邻扫描行中心跳变容忍比例。"""
    center_jump_ratio: float = 0.12
    """相邻扫描行宽度跳变容忍比例。"""
    width_jump_ratio: float = 0.45
    """拟合外点剔除阈值像素。"""
    fit_outlier_threshold_px: float = 28.0
    """最小可接受通道宽度比例。"""
    min_corridor_width_ratio: float = 0.12
    """BEV 预览图输出后缀。"""
    bev_overlay_suffix: str = "_bev_overlay.png"
    """导航线输出后缀。"""
    overlay_suffix: str = "_nav_overlay.png"
    """导航数据输出后缀。"""
    json_suffix: str = "_nav.json"


@dataclass
class AutoLabelConfig:
    """自动打标配置。"""

    """标签名称。"""
    label_name: str = "id1"
    """轮廓最小面积。"""
    min_area: float = 64.0
    """轮廓简化比例。"""
    approx_epsilon_ratio: float = 0.002
    """是否写入空标注。"""
    write_empty_annotations: bool = False
    """是否同时导出覆盖图用于复核。"""
    save_overlay: bool = True
    """自动打标覆盖图后缀。"""
    overlay_suffix: str = "_autolabel_overlay.png"
    """Labelme version 字段。"""
    json_version: str = "5.0.1"


@dataclass
class WebConfig:
    """Web 操作台配置。"""

    """Web 服务监听地址。"""
    host: str = "127.0.0.1"
    """Web 服务监听端口。"""
    port: int = 5378
    """页面标题。"""
    title: str = "CaneSegLab 操作台"
    """最近训练记录展示数量。"""
    recent_runs_limit: int = 6
    """是否限制同一时间只运行一个后台任务。"""
    single_job_mode: bool = True


@dataclass
class PaperFigureTemplateConfig:
    """论文图模板配置。"""

    width_in: float = 6.4
    height_in: float = 4.8
    line_width: float | None = None
    legend_loc: str = "best"
    show_grid: bool = True
    x_rotation: int = 0
    y_rotation: int = 0
    bins: int = 20
    panel_spacing: float = 0.1
    image_border: bool = False
    caption_enabled: bool = False


@dataclass
class PaperFigureConfig:
    """论文图导出配置。"""

    """论文图默认输出目录。"""
    output_dir: Path = Path("data/paper_figures")
    """导出文件格式。"""
    formats: tuple[str, ...] = ("png",)
    """导出分辨率。"""
    dpi: int = 300
    """是否导出透明背景。"""
    transparent: bool = True
    """保存时的边界裁切方式。"""
    bbox_inches: str = "tight"
    """保存时图像边缘留白。"""
    pad_inches: float = 0.08
    """默认图宽，单位英寸。"""
    default_width_in: float = 4.8
    """默认图高，单位英寸。"""
    default_height_in: float = 4.8
    """中文字体候选列表。"""
    font_family_cn: tuple[str, ...] = (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    )
    """英文字体候选列表。"""
    font_family_en: tuple[str, ...] = (
        "Times New Roman",
        "DejaVu Serif",
    )
    """全局默认字号。"""
    font_size: int = 14
    """标题字号。"""
    title_size: int = 12
    """坐标轴标签字号。"""
    label_size: int = 14
    """坐标轴刻度字号。"""
    tick_size: int = 14
    """图例字号。"""
    legend_size: int = 14
    """图中标注文字字号。"""
    annotation_size: int = 14
    """折线默认线宽。"""
    line_width: float = 2.5
    """标记点默认大小。"""
    marker_size: float = 5.0
    """是否显示网格。"""
    grid_enabled: bool = True
    """网格透明度。"""
    grid_alpha: float = 0.25
    """坐标轴边框线宽。"""
    spine_width: float = 1.2
    """默认配色循环。"""
    color_cycle: tuple[str, ...] = (
        "#1F4E79",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#4D4D4D",
    )
    """画布背景色。"""
    background_color: str = "none"
    """坐标轴区域背景色。"""
    axes_facecolor: str = "none"
    """保存图片时使用的背景色。"""
    save_facecolor: str = "none"
    """是否自动紧凑排版。"""
    tight_layout: bool = True
    """是否修复负号显示问题。"""
    minus_sign_fix: bool = True
    """各类图模板的局部配置。"""
    templates: dict[str, PaperFigureTemplateConfig] = field(
        default_factory=_default_paper_figure_templates,
    )


@dataclass
class BenchmarkConfig:
    """推理测速配置。"""

    output_dir: Path = Path("data/benchmarks")
    warmup_runs: int = 5
    timed_runs: int = 30
    max_images: int = 12
    backends: tuple[str, ...] = ("pytorch", "onnx", "tensorrt")


@dataclass
class ProjectConfig:
    """
    项目总配置对象。

    说明：
    - 所有子系统配置统一从该对象访问
    - 入口脚本仅需构造并传入该对象
    """

    """数据集配置。"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """标注转掩码配置。"""
    mask: MaskConfig = field(default_factory=MaskConfig)
    """训练配置。"""
    train: TrainConfig = field(default_factory=TrainConfig)
    """推理配置。"""
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    """导航线拟合配置。"""
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    """自动打标配置。"""
    auto_label: AutoLabelConfig = field(default_factory=AutoLabelConfig)
    """Web 配置。"""
    web: WebConfig = field(default_factory=WebConfig)
    """论文图配置。"""
    paper_figures: PaperFigureConfig = field(default_factory=PaperFigureConfig)
    """测速配置。"""
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
