from __future__ import annotations

import copy
import gc
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Sequence

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..artifacts import create_run_dir, resolve_artifact_path
from ..config import ProjectConfig
from ..model import SegmentationModel
from .log_service import logger


# ------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------


@dataclass(frozen=True)
class SegmentationSample:
    """
    表示一个训练样本

    image_path
        原始图片路径

    mask_path
        对应标注mask路径
    """

    image_path: Path
    mask_path: Path


@dataclass
class EpochResult:
    """
    每个 epoch 的训练结果
    """

    loss: float
    dice: float
    iou: float


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------


class SegmentationDataset(Dataset):
    """
    语义分割数据集

    负责：
    - 读取图片
    - 读取mask
    - 数据增强
    - 转换为tensor
    """

    def __init__(
        self,
        samples: Sequence[SegmentationSample],
        transforms=None,
        num_classes: int = 1,
        binary_threshold: int = 127,
    ):
        self.samples = list(samples)
        self.transforms = transforms
        self.num_classes = num_classes
        self.binary_threshold = binary_threshold

        logger.info("数据集已加载：%d 个样本", len(self.samples))

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index):
        """读取一个样本"""

        sample = self.samples[index]

        # 读取图像
        image = self._read_image(sample.image_path)

        # 读取mask
        mask = self._read_mask(sample.mask_path)

        # 数据增强
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # HWC -> CHW (PyTorch格式)
        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.from_numpy(image).float()

        mask_tensor = torch.from_numpy(self._encode_mask(mask)).float()

        return image_tensor, mask_tensor

    @staticmethod
    def _read_image(path: Path):
        """读取图像"""

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(path)

        return image.astype(np.float32) / 255.0

    @staticmethod
    def _read_mask(path: Path):
        """读取mask"""

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(path)

        return mask

    def _encode_mask(self, mask: np.ndarray):
        """
        将mask转换为训练格式
        """

        # 单类别分割（二值）
        if self.num_classes == 1:
            binary_mask = (mask > self.binary_threshold).astype(np.float32)
            return np.expand_dims(binary_mask, axis=0)

        # 多类别 one-hot
        height, width = mask.shape

        one_hot = np.zeros((self.num_classes, height, width), dtype=np.float32)

        for i in range(self.num_classes):
            one_hot[i] = (mask == i).astype(np.float32)

        return one_hot


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss

    BCE 关注像素分类
    Dice 关注区域重叠
    """

    def __init__(self, dice_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)

        numerator = 2 * (probs * targets).sum(dim=(2, 3))
        denominator = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-6

        dice = (numerator / denominator).mean()

        return bce + self.dice_weight * (1 - dice)


# ------------------------------------------------------------
# TrainingService
# ------------------------------------------------------------


class TrainingService:
    """
    训练服务

    负责：
    - 数据扫描
    - 数据加载
    - 模型训练
    - 模型验证
    """

    def __init__(self, config: ProjectConfig):
        self.project_cfg = config
        self.cfg = config.train
        self.output_dir = self.cfg.output_dir


        # 设备
        self.device = self._resolve_device()

        logger.info("训练设备：%s", self.device)

        # 设置随机种子
        self.set_global_seed(self.cfg.random_seed)
        logger.info("随机种子：%d", self.cfg.random_seed)

        # 损失函数
        self.criterion = BCEDiceLoss(self.cfg.dice_loss_weight)

        # 数据增强
        self.train_transform = self._build_train_transform()
        self.valid_transform = self._build_valid_transform()

    # ------------------------------------------------------------

    def train(self, image_dir: Path, mask_dir: Path):
        """
        训练入口
        """

        started_at = datetime.now()
        train_started = perf_counter()
        logger.info("开始扫描数据集：图像目录=%s，掩码目录=%s", image_dir, mask_dir)

        samples = self._build_samples(image_dir, mask_dir)

        if not samples:
            raise RuntimeError("未扫描到可用样本")

        self.output_dir = create_run_dir(self.cfg.output_dir)
        logger.info("本次训练输出目录：%s", self.output_dir)

        logger.info("数据集扫描完成：共 %d 个样本", len(samples))

        train_samples, valid_samples, test_samples = self._split_samples(samples)

        logger.info(
            "数据集划分完成：训练集=%d，验证集=%d，测试集=%d",
            len(train_samples),
            len(valid_samples),
            len(test_samples),
        )

        train_loader = self._build_dataloader(train_samples, True)
        valid_loader = self._build_dataloader(valid_samples, False)
        logger.info("数据加载器构建完成：训练批次=%d，验证批次=%d", len(train_loader), len(valid_loader))

        model = self._build_model()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        use_amp = self.cfg.use_amp and self.device.type == "cuda"
        scaler = amp.GradScaler(enabled=use_amp)  # type: ignore
        logger.info(
            "训练准备完成：输出目录=%s，AMP=%s，梯度累积=%d",
            self.output_dir,
            "开启" if use_amp else "关闭",
            self.cfg.grad_accum_steps,
        )

        history = self._fit(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            grad_accum_steps=self.cfg.grad_accum_steps,
            use_amp=use_amp,
            history_context=self._build_history_context(
                image_dir=image_dir,
                mask_dir=mask_dir,
                train_samples=train_samples,
                valid_samples=valid_samples,
                test_samples=test_samples,
                started_at=started_at,
                use_amp=use_amp,
            ),
        )

        self._save_history(history)
        if self.cfg.export_onnx:
            onnx_path = self._export_onnx_model(model)
            history["artifacts"]["onnx_path"] = str(onnx_path.resolve()) if onnx_path else None
            self._save_history(history)

        finished_at = datetime.now()
        history["meta"]["finished_at"] = finished_at.isoformat(timespec="seconds")
        history["meta"]["duration_sec"] = round(perf_counter() - train_started, 3)
        self._save_history(history)

        return model, history

    def export_onnx(self, onnx_output_path: Path | None = None) -> Path:
        """导出 ONNX 模型。"""

        from onnx import load_model
        from onnx.checker import check_model

        output_path = onnx_output_path or self._artifact_path("model.onnx")
        weight_path = self._artifact_path("best_model.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("开始导出 ONNX：权重=%s，输出=%s", weight_path, output_path)

        model = self._build_model()
        state_dict = torch.load(str(weight_path), map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        dummy_input = torch.randn(
            1,
            self.cfg.in_channels,
            self.cfg.crop_height,
            self.cfg.crop_width,
            device=self.device,
        )

        dynamic_axes = None
        if self.cfg.onnx_dynamic_batch:
            dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=self.cfg.onnx_opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
            )

        onnx_model = load_model(str(output_path))
        check_model(onnx_model)
        logger.info("ONNX 导出完成：%s", output_path)
        return output_path

    # ------------------------------------------------------------

    def _build_samples(
        self, image_dir: Path, mask_dir: Path
    ) -> list[SegmentationSample]:
        """
        扫描数据集并构建样本列表
        """

        samples = []

        iterator = image_dir.iterdir()

        for path in iterator:
            if not path.is_file():
                continue

            if path.suffix.lower() not in self.cfg.image_extensions:
                continue

            mask_name = f"{path.stem}{self.cfg.mask_suffix}"
            mask_path = mask_dir / mask_name

            if mask_path.exists():
                samples.append(SegmentationSample(path, mask_path))

        return samples

    # ------------------------------------------------------------

    def _split_samples(self, samples: list[SegmentationSample]):
        """
        划分训练集、验证集与测试集
        """

        samples = list(samples)

        random.Random(self.cfg.random_seed).shuffle(samples)

        if self.cfg.train_ratio <= 0 or self.cfg.valid_ratio < 0:
            raise ValueError("数据集划分比例必须为非负，且训练集比例必须大于 0")
        if self.cfg.train_ratio + self.cfg.valid_ratio >= 1:
            raise ValueError("训练集比例与验证集比例之和必须小于 1，以保留测试集")

        total = len(samples)
        train_end = max(1, int(total * self.cfg.train_ratio))
        valid_end = train_end + int(total * self.cfg.valid_ratio)
        valid_end = min(valid_end, total)

        train_samples = samples[:train_end]
        valid_samples = samples[train_end:valid_end]
        test_samples = samples[valid_end:]

        if not valid_samples and test_samples:
            valid_samples, test_samples = test_samples[:1], test_samples[1:]

        return train_samples, valid_samples, test_samples

    # ------------------------------------------------------------

    def _build_dataloader(self, samples: list[SegmentationSample], train: bool):
        """
        构建 DataLoader
        """

        dataset = SegmentationDataset(
            samples,
            self.train_transform if train else self.valid_transform,
            self.cfg.num_classes,
            self.cfg.binary_threshold,
        )

        return DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size if train else self.cfg.valid_batch_size,
            shuffle=train,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    # ------------------------------------------------------------

    def _build_model(self):
        """
        构建 segmentation model
        """

        logger.info(
            "开始构建模型：%s，backbone=%s，输入通道=%d，类别数=%d",
            self.cfg.model_name,
            self.cfg.backbone,
            self.cfg.in_channels,
            self.cfg.num_classes,
        )
        return SegmentationModel.from_train_config(self.cfg).to(self.device)

    # ------------------------------------------------------------

    def _build_optimizer(self, model):
        """
        构建优化器
        """

        if self.cfg.optimizer_name.lower() != "adam":
            raise ValueError(f"暂不支持的优化器: {self.cfg.optimizer_name}")
        logger.info(
            "优化器：%s，学习率=%.6f，权重衰减=%.6f",
            self.cfg.optimizer_name,
            self.cfg.learning_rate,
            self.cfg.weight_decay,
        )
        return optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    # ------------------------------------------------------------

    def _build_scheduler(self, optimizer):
        """
        构建学习率调度器
        """

        name = self.cfg.scheduler_name.lower()

        if name == "none":
            logger.info("学习率调度器：未启用")
            return None

        if name == "cosine":
            logger.info("学习率调度器：CosineAnnealingLR")
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.epochs,
                eta_min=self.cfg.min_learning_rate,
            )

        if name == "plateau":
            logger.info("学习率调度器：ReduceLROnPlateau")
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.cfg.scheduler_patience,
                min_lr=self.cfg.min_learning_rate,
            )

        raise ValueError(name)

    # ------------------------------------------------------------

    def _train_one_epoch(
        self,
        model,
        optimizer,
        scaler,
        dataloader,
        grad_accum_steps: int,
        use_amp: bool,
    ) -> EpochResult:

        grad_accum_steps = max(1, grad_accum_steps)

        model.train()

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0
        total_dice = 0
        total_iou = 0
        count = 0

        progress = tqdm(dataloader, desc="训练")

        for step, (images, masks) in enumerate(progress):
            images = images.to(self.device)
            masks = masks.to(self.device)

            batch = images.size(0)

            # 自动混合精度
            with amp.autocast(enabled=use_amp):  # type: ignore
                logits = model(images)

                loss = self.criterion(logits, masks)

                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad(set_to_none=True)

            probs = torch.sigmoid(logits).detach()

            dice = self._dice_score(masks, probs, self.cfg.metric_threshold)
            iou = self._iou_score(masks, probs, self.cfg.metric_threshold)

            total_loss += loss.item() * batch
            total_dice += dice * batch
            total_iou += iou * batch
            count += batch

            progress.set_postfix(
                loss=f"{total_loss / count:.4f}",
                dice=f"{total_dice / count:.4f}",
                iou=f"{total_iou / count:.4f}",
            )

        return EpochResult(
            total_loss / count,
            total_dice / count,
            total_iou / count,
        )

    # ------------------------------------------------------------

    @torch.no_grad()
    def _validate_one_epoch(self, model, dataloader) -> EpochResult:

        model.eval()

        total_loss = 0
        total_dice = 0
        total_iou = 0
        count = 0

        progress = tqdm(dataloader, desc="验证")

        for images, masks in progress:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = model(images)

            loss = self.criterion(logits, masks)

            probs = torch.sigmoid(logits)

            batch = images.size(0)

            dice = self._dice_score(masks, probs, self.cfg.metric_threshold)
            iou = self._iou_score(masks, probs, self.cfg.metric_threshold)

            total_loss += loss.item() * batch
            total_dice += dice * batch
            total_iou += iou * batch
            count += batch

        return EpochResult(
            total_loss / count,
            total_dice / count,
            total_iou / count,
        )

    # ------------------------------------------------------------

    def _fit(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        valid_loader,
        grad_accum_steps: int,
        use_amp: bool,
        history_context: dict[str, object],
    ):

        best_dice = -1
        best_weight = None
        best_path = self.output_dir / "best_model.pt"

        history = {
            "meta": dict(history_context["meta"]),
            "dataset": dict(history_context["dataset"]),
            "training": dict(history_context["training"]),
            "epochs": [],
            "artifacts": {
                "history_path": str((self.output_dir / "history.json").resolve()),
                "best_model_path": str(best_path.resolve()),
                "onnx_path": None,
            },
            "train_loss": [],
            "valid_loss": [],
            "valid_dice": [],
            "valid_iou": [],
            "learning_rate": [],
            "epoch_time_sec": [],
        }

        logger.info("开始训练：共 %d 轮", self.cfg.epochs)

        for epoch in range(self.cfg.epochs):
            logger.info("开始第 %d/%d 轮训练", epoch + 1, self.cfg.epochs)

            gc.collect()
            epoch_started = perf_counter()

            train_result = self._train_one_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                dataloader=train_loader,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
            )

            valid_result = self._validate_one_epoch(
                model=model,
                dataloader=valid_loader,
            )

            history["train_loss"].append(train_result.loss)
            history["valid_loss"].append(valid_result.loss)
            history["valid_dice"].append(valid_result.dice)
            history["valid_iou"].append(valid_result.iou)

            self._step_scheduler(scheduler, valid_result.loss)
            current_lr = self._current_lr(optimizer)
            epoch_time_sec = round(perf_counter() - epoch_started, 3)
            history["learning_rate"].append(current_lr)
            history["epoch_time_sec"].append(epoch_time_sec)
            history["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_result.loss, 6),
                    "valid_loss": round(valid_result.loss, 6),
                    "valid_dice": round(valid_result.dice, 6),
                    "valid_iou": round(valid_result.iou, 6),
                    "lr": round(current_lr, 10),
                    "epoch_time_sec": epoch_time_sec,
                }
            )

            if valid_result.dice > best_dice:
                previous_best = best_dice
                best_dice = valid_result.dice
                best_weight = copy.deepcopy(model.state_dict())
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_weight, best_path)
                history["artifacts"]["best_epoch"] = epoch + 1
                history["artifacts"]["best_valid_dice"] = round(valid_result.dice, 6)
                history["artifacts"]["best_valid_loss"] = round(valid_result.loss, 6)
                logger.info(
                    "最佳模型已更新：Dice %.4f -> %.4f，已保存到 %s",
                    previous_best if previous_best >= 0 else 0.0,
                    best_dice,
                    best_path,
                )

            history["training"]["epochs_completed"] = epoch + 1

            logger.info(
                "第 %d/%d 轮完成：训练损失=%.4f，验证损失=%.4f，验证 Dice=%.4f，验证 IoU=%.4f，学习率=%.6f",
                epoch + 1,
                self.cfg.epochs,
                train_result.loss,
                valid_result.loss,
                valid_result.dice,
                valid_result.iou,
                current_lr,
            )

        model.load_state_dict(best_weight)
        logger.info("训练结束：最佳验证 Dice=%.4f，最佳权重=%s", best_dice, best_path)

        return history

    def _save_history(self, history: dict[str, object]) -> Path:
        output_path = self.output_dir / "history.json"
        output_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("训练历史已保存：%s", output_path)
        return output_path

    def _export_onnx_model(self, model) -> Path | None:
        output_path = self.output_dir / "model.onnx"
        try:
            logger.info("开始导出本次训练的 ONNX：%s", output_path)
            return self.export_onnx(output_path)
        except Exception as exc:
            logger.warning("ONNX 导出失败，但训练权重已保存：%s", exc)
            return None

    def _artifact_path(self, filename: str) -> Path:
        current_path = self.output_dir / filename
        if self.output_dir != self.cfg.output_dir and current_path.exists():
            return current_path
        return resolve_artifact_path(self.cfg.output_dir, filename)

    def _build_history_context(
        self,
        image_dir: Path,
        mask_dir: Path,
        train_samples: list[SegmentationSample],
        valid_samples: list[SegmentationSample],
        test_samples: list[SegmentationSample],
        started_at: datetime,
        use_amp: bool,
    ) -> dict[str, object]:
        return {
            "meta": {
                "schema_version": 2,
                "run_name": self.output_dir.name,
                "created_at": started_at.isoformat(timespec="seconds"),
                "finished_at": None,
                "duration_sec": None,
            },
            "dataset": {
                "image_dir": str(image_dir.resolve()),
                "mask_dir": str(mask_dir.resolve()),
                "total_samples": len(train_samples) + len(valid_samples) + len(test_samples),
                "train_samples": len(train_samples),
                "valid_samples": len(valid_samples),
                "test_samples": len(test_samples),
                "train_ratio": self.cfg.train_ratio,
                "valid_ratio": self.cfg.valid_ratio,
                "test_ratio": round(max(0.0, 1.0 - self.cfg.train_ratio - self.cfg.valid_ratio), 6),
                "image_extensions": list(self.cfg.image_extensions),
                "mask_suffix": self.cfg.mask_suffix,
            },
            "training": {
                "model_name": self.cfg.model_name,
                "backbone": self.cfg.backbone,
                "in_channels": self.cfg.in_channels,
                "num_classes": self.cfg.num_classes,
                "epochs_planned": self.cfg.epochs,
                "epochs_completed": 0,
                "train_batch_size": self.cfg.train_batch_size,
                "valid_batch_size": self.cfg.valid_batch_size,
                "learning_rate": self.cfg.learning_rate,
                "min_learning_rate": self.cfg.min_learning_rate,
                "weight_decay": self.cfg.weight_decay,
                "optimizer_name": self.cfg.optimizer_name,
                "scheduler_name": self.cfg.scheduler_name,
                "grad_accum_steps": self.cfg.grad_accum_steps,
                "use_amp": use_amp,
                "crop_width": self.cfg.crop_width,
                "crop_height": self.cfg.crop_height,
                "metric_threshold": self.cfg.metric_threshold,
                "random_seed": self.cfg.random_seed,
                "device": str(self.device),
            },
        }

    # ------------------------------------------------------------

    def _build_train_transform(self):
        transforms = [
            A.Resize(
                width=self.cfg.crop_width,
                height=self.cfg.crop_height,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            )
        ]
        if self.cfg.enable_augmentation:
            transforms = [
                A.HorizontalFlip(p=self.cfg.horizontal_flip_prob),
                A.RandomBrightnessContrast(p=self.cfg.brightness_contrast_prob),
                *transforms,
            ]
        return A.Compose(transforms) # type: ignore

    def _build_valid_transform(self):

        return A.Compose(
            [
                A.Resize(
                    width=self.cfg.crop_width,
                    height=self.cfg.crop_height,
                    interpolation=cv2.INTER_AREA,
                    mask_interpolation=cv2.INTER_NEAREST,
                )
            ]
        )

    # ------------------------------------------------------------

    @staticmethod
    def set_global_seed(seed):
        """固定随机种子"""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_device(self):
        device_name = self.cfg.device.strip().lower()
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("配置要求使用 %s，但当前 CUDA 不可用，已回退到 CPU", self.cfg.device)
            return torch.device("cpu")
        return torch.device(self.cfg.device)

    @staticmethod
    def _step_scheduler(scheduler, valid_loss: float) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)
            return
        scheduler.step()

    @staticmethod
    def _current_lr(optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------

    @staticmethod
    def _dice_score(targets, probs, threshold=0.5):
        """计算 Dice"""

        target = (targets > threshold).float()
        pred = (probs > threshold).float()

        intersection = (target * pred).sum(dim=(2, 3))

        union = target.sum(dim=(2, 3)) + pred.sum(dim=(2, 3))

        dice = (2 * intersection + 1e-6) / (union + 1e-6)

        return float(dice.mean().item())

    @staticmethod
    def _iou_score(targets, probs, threshold=0.5):
        """计算 IoU"""

        target = (targets > threshold).float()
        pred = (probs > threshold).float()

        intersection = (target * pred).sum(dim=(2, 3))

        union = target.sum(dim=(2, 3)) + pred.sum(dim=(2, 3)) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)

        return float(iou.mean().item())
