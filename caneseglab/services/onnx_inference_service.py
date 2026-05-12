from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort

from ..artifacts import resolve_artifact_path
from ..config import ProjectConfig
from .inference_base_service import InferenceBaseService
from .log_service import logger


class OnnxInferenceService(InferenceBaseService):
    """ONNX 推理服务。"""

    _session_cache: dict[
        tuple[str, int, int],
        tuple[ort.InferenceSession, str, tuple[int, int] | None],
    ] = {}
    _session_cache_lock = Lock()

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__(config)

        self._onnx_session: ort.InferenceSession | None = None
        self._onnx_input_name: str | None = None
        self._onnx_input_hw: tuple[int, int] | None = None
        self._onnx_path: Path | None = None

        try:
            self._onnx_path = resolve_artifact_path(
                self.cfg.artifacts_dir,
                "model.onnx",
                self.cfg.onnx_path,
            )
        except FileNotFoundError:
            self._onnx_path = self.cfg.onnx_path

        if self._onnx_path and self._onnx_path.exists():
            (
                self._onnx_session,
                self._onnx_input_name,
                self._onnx_input_hw,
            ) = self._get_cached_session(self._onnx_path)

    @classmethod
    def _cache_key(cls, model_path: Path) -> tuple[str, int, int]:
        resolved = model_path.resolve()
        stat = resolved.stat()
        return (str(resolved), stat.st_mtime_ns, stat.st_size)

    @classmethod
    def _get_cached_session(
        cls,
        model_path: Path,
    ) -> tuple[ort.InferenceSession, str, tuple[int, int] | None]:
        cache_key = cls._cache_key(model_path)

        with cls._session_cache_lock:
            cached = cls._session_cache.get(cache_key)
            if cached is not None:
                logger.info("复用已加载 ONNX 模型：%s", cache_key[0])
                return cached

            available = set(ort.get_available_providers())
            providers = [
                name
                for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if name in available
            ]
            session = ort.InferenceSession(
                cache_key[0],
                providers=providers,
            )
            input_meta = session.get_inputs()[0]
            input_name = input_meta.name

            input_hw: tuple[int, int] | None = None
            shape = input_meta.shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                input_hw = (shape[2], shape[3])

            cached = (session, input_name, input_hw)
            cls._session_cache = {
                key: value
                for key, value in cls._session_cache.items()
                if key[0] != cache_key[0]
            }
            cls._session_cache[cache_key] = cached
            logger.info("已加载 ONNX 模型：%s", cache_key[0])
            return cached

    def infer_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._onnx_session is None:
            raise RuntimeError("ONNX 模型未加载，请先训练或导出 ONNX 模型")

        tensor, working = self._preprocess(image, input_hw=self._onnx_input_hw)
        output = self._onnx_session.run(None, {self._onnx_input_name: tensor})[0]

        mask = self._logits_to_mask(output)
        overlay = self._build_overlay(working, mask)
        return mask, overlay

    def infer_file(self, path: Path) -> tuple[Path, Path]:
        image = self._load_image(path)
        mask, overlay = self.infer_image(image)
        return self._save_result(path, mask, overlay, "onnx")

    # backward-compatible aliases
    def infer_onnx_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.infer_image(image)

    def infer_onnx_file(self, path: Path) -> tuple[Path, Path]:
        return self.infer_file(path)
