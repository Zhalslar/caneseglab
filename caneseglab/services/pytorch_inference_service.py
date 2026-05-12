from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..artifacts import resolve_artifact_path
from ..config import ProjectConfig
from ..model import SegmentationModel
from .inference_base_service import InferenceBaseService


class PytorchInferenceService(InferenceBaseService):
    """PyTorch 推理服务。"""

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__(config)
        self._project_cfg = config
        self.train_cfg = config.train
        self.device = self._resolve_device()
        self.weight_path = resolve_artifact_path(
            self.train_cfg.output_dir,
            "best_model.pt",
        )
        self.model = self._load_model()
        self._input_hw = (self.train_cfg.crop_height, self.train_cfg.crop_width)

    def infer_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, working = self._preprocess(image, input_hw=self._input_hw)
        input_tensor = torch.from_numpy(tensor).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor).detach().cpu().numpy()

        mask = self._logits_to_mask(logits)
        overlay = self._build_overlay(working, mask)
        return mask, overlay

    def infer_file(self, path: Path) -> tuple[Path, Path]:
        image = self._load_image(path)
        mask, overlay = self.infer_image(image)
        return self._save_result(path, mask, overlay, "torch")

    def _load_model(self):
        model = SegmentationModel.from_train_config(self.train_cfg).to(self.device)
        state_dict = torch.load(str(self.weight_path), map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _resolve_device(self) -> torch.device:
        device_name = self.train_cfg.device.strip().lower()
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.train_cfg.device)
