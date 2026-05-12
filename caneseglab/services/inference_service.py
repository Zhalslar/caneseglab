from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "InferenceBaseService",
    "OnnxInferenceService",
    "TensorRTInferenceService",
]

if TYPE_CHECKING:
    from .inference_base_service import InferenceBaseService
    from .onnx_inference_service import OnnxInferenceService
    from .tensorrt_inference_service import TensorRTInferenceService


def __getattr__(name: str):
    # Compatibility layer with lazy import to avoid cross-backend dependency coupling.
    if name == "InferenceBaseService":
        from .inference_base_service import InferenceBaseService

        return InferenceBaseService
    if name == "OnnxInferenceService":
        from .onnx_inference_service import OnnxInferenceService

        return OnnxInferenceService
    if name == "TensorRTInferenceService":
        from .tensorrt_inference_service import TensorRTInferenceService

        return TensorRTInferenceService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
