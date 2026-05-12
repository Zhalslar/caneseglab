from __future__ import annotations

import os
import subprocess
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from ..artifacts import resolve_artifact_path
from ..config import ProjectConfig
from .inference_base_service import InferenceBaseService
from .log_service import logger

if TYPE_CHECKING:
    import tensorrt as trt


class TensorRTInferenceService(InferenceBaseService):
    """TensorRT 推理服务。"""

    _engine_cache: dict[tuple[str, int, int], Any] = {}
    _engine_cache_lock = Lock()
    _dll_runtime_configured = False
    _dll_directory_handles: list[Any] = []

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__(config)
        self.engine = None
        self._context = None
        self._context_engine_id: int | None = None
        self._trt: Any | None = None
        self._trt_logger: Any | None = None

    @staticmethod
    def _prepend_path(path: Path) -> None:
        resolved = str(path.resolve())
        current = os.environ.get("PATH", "")
        parts = [item for item in current.split(os.pathsep) if item]
        if resolved in parts:
            return
        os.environ["PATH"] = (
            f"{resolved}{os.pathsep}{current}" if current else resolved
        )

    def _candidate_tensorrt_roots(self) -> list[Path]:
        roots: list[Path] = []
        env_home = os.environ.get("TENSORRT_HOME")
        if env_home:
            roots.append(Path(env_home))

        repo_root = Path(__file__).resolve().parents[2]
        roots.extend(sorted(repo_root.glob("TensorRT-*"), reverse=True))
        return roots

    def _configure_tensorrt_runtime(self) -> None:
        cls = type(self)
        if cls._dll_runtime_configured:
            return

        for root in self._candidate_tensorrt_roots():
            bin_dir = root / "bin"
            lib_dir = root / "lib"
            if not (bin_dir / "nvinfer_10.dll").exists():
                continue

            os.environ.setdefault("TENSORRT_HOME", str(root.resolve()))
            for path in (lib_dir, bin_dir):
                if not path.exists():
                    continue
                self._prepend_path(path)
                if hasattr(os, "add_dll_directory"):
                    cls._dll_directory_handles.append(
                        os.add_dll_directory(str(path.resolve()))
                    )

            cls._dll_runtime_configured = True
            logger.info("Detected local TensorRT runtime: %s", root.resolve())
            return

    def _get_trt(self):
        if self._trt is None:
            self._configure_tensorrt_runtime()
            try:
                import tensorrt as trt
            except Exception as exc:
                raise RuntimeError(
                    "TensorRT 推理依赖 tensorrt。"
                    "请确认已正确安装，并且仅在需要 TensorRT 推理时再初始化该服务。"
                ) from exc
            self._trt = trt
        return self._trt

    def _get_trt_logger(self):
        if self._trt_logger is None:
            trt = self._get_trt()

            class _TensorRTLogger(trt.ILogger):  # type: ignore[misc, valid-type]
                def __init__(self) -> None:
                    trt.ILogger.__init__(self)

                def log(self, severity, message) -> None:
                    text = str(message).rstrip()
                    if not text:
                        return

                    if severity == trt.ILogger.INTERNAL_ERROR:
                        logger.error("TensorRT: %s", text)
                    elif severity == trt.ILogger.ERROR:
                        logger.error("TensorRT: %s", text)
                    elif severity == trt.ILogger.WARNING:
                        logger.warning("TensorRT: %s", text)
                    elif severity == trt.ILogger.INFO:
                        logger.info("TensorRT: %s", text)
                    else:
                        logger.debug("TensorRT: %s", text)

            self._trt_logger = _TensorRTLogger()

        return self._trt_logger

    def _resolve_onnx_path(self, onnx_path: Path | None = None) -> Path:
        return resolve_artifact_path(
            self.cfg.artifacts_dir,
            "model.onnx",
            onnx_path or self.cfg.onnx_path,
        )

    def _resolve_trtexec_path(self) -> Path:
        self._configure_tensorrt_runtime()
        for root in self._candidate_tensorrt_roots():
            candidate = root / "bin" / "trtexec.exe"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "未找到 trtexec.exe，请确认 TensorRT Windows ZIP 已解压到项目根目录，"
            "或已正确设置 TENSORRT_HOME。"
        )

    def _get_onnx_input_profile(self, onnx_path: Path) -> tuple[str, str]:
        from onnx import load_model

        model = load_model(str(onnx_path))
        if not model.graph.input:
            raise RuntimeError(f"ONNX 模型缺少输入信息：{onnx_path}")

        input_tensor = model.graph.input[0]
        dims = input_tensor.type.tensor_type.shape.dim
        if len(dims) != 4:
            raise RuntimeError(
                f"暂不支持非 4 维输入的 ONNX：{onnx_path} / {input_tensor.name}"
            )

        shape_values: list[int] = []
        fallbacks = (
            1,
            int(self._project_cfg.train.in_channels),
            int(self._project_cfg.train.crop_height),
            int(self._project_cfg.train.crop_width),
        )
        for dim, fallback in zip(dims, fallbacks, strict=True):
            value = int(dim.dim_value) if getattr(dim, "dim_value", 0) else fallback
            shape_values.append(value)

        shape_text = "x".join(str(item) for item in shape_values)
        return input_tensor.name or "input", shape_text

    def export_engine(
        self,
        onnx_path: Path | None = None,
        engine_output_path: Path | None = None,
    ) -> Path:
        source_onnx = self._resolve_onnx_path(onnx_path)
        if not source_onnx.exists():
            raise FileNotFoundError(source_onnx)

        output_path = engine_output_path or source_onnx.with_name("model.trt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trtexec_path = self._resolve_trtexec_path()
        input_name, input_shape = self._get_onnx_input_profile(source_onnx)
        command = [
            str(trtexec_path),
            f"--onnx={source_onnx}",
            f"--saveEngine={output_path}",
            f"--shapes={input_name}:{input_shape}",
            "--skipInference",
        ]

        logger.info(
            "开始导出 TensorRT：onnx=%s，engine=%s，shape=%s:%s",
            source_onnx,
            output_path,
            input_name,
            input_shape,
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            tail = "\n".join(
                line for line in (completed.stdout + "\n" + completed.stderr).splitlines()[-20:]
            )
            raise RuntimeError(
                "TensorRT 引擎导出失败，请检查 ONNX 模型和 TensorRT 环境。\n"
                f"命令：{' '.join(command)}\n"
                f"输出片段：\n{tail}"
            )

        logger.info("TensorRT 引擎导出完成：%s", output_path)
        return output_path.resolve()

    def _decode_cuda_error(self, cudart, code) -> tuple[str, str]:
        name = str(code)
        detail = ""
        try:
            err_name = cudart.cudaGetErrorName(code)
            if isinstance(err_name, tuple) and len(err_name) > 1:
                raw = err_name[1]
                name = raw.decode() if isinstance(raw, bytes) else str(raw)

            err_text = cudart.cudaGetErrorString(code)
            if isinstance(err_text, tuple) and len(err_text) > 1:
                raw = err_text[1]
                detail = raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception:
            pass
        return name, detail

    def _ensure_cuda_ready(self) -> None:
        try:
            from cuda.bindings import runtime as cudart
        except Exception as exc:
            raise RuntimeError(
                "TensorRT 推理依赖 cuda-python (cuda.bindings)。"
                "请确认已正确安装并且当前虚拟环境可导入。"
            ) from exc

        result = cudart.cudaGetDeviceCount()
        if isinstance(result, tuple):
            code = result[0]
            count = int(result[1]) if len(result) > 1 else 0
        else:
            code = result
            count = 0

        if code != cudart.cudaError_t.cudaSuccess:
            name, detail = self._decode_cuda_error(cudart, code)
            msg = f"CUDA 初始化失败: {name}"
            if detail:
                msg = f"{msg} ({detail})"
            raise RuntimeError(
                f"{msg}。请检查显卡驱动、CUDA 运行时与 TensorRT 版本是否匹配。"
            )

        if count < 1:
            raise RuntimeError("未检测到可用 CUDA 设备，无法执行 TensorRT 推理。")

    def _collect_cuda_diagnostics(self) -> list[str]:
        try:
            trt = self._get_trt()
            details = [f"tensorrt={getattr(trt, '__version__', 'unknown')}"]
        except Exception as exc:
            details = [f"tensorrt_import_error={exc}"]

        try:
            import torch

            details.append(f"torch={torch.__version__}")
            details.append(f"torch.cuda={torch.version.cuda}")
            available = torch.cuda.is_available()
            details.append(f"torch.cuda.is_available={available}")
            if available:
                details.append(f"torch.cuda.device_count={torch.cuda.device_count()}")
        except Exception as exc:
            details.append(f"torch_probe_error={exc}")

        try:
            from cuda.bindings import runtime as cudart

            result = cudart.cudaGetDeviceCount()
            if isinstance(result, tuple):
                code = result[0]
                count = int(result[1]) if len(result) > 1 else 0
            else:
                code = result
                count = 0

            if code == cudart.cudaError_t.cudaSuccess:
                details.append(f"cudaGetDeviceCount={count}")
            else:
                name, text = self._decode_cuda_error(cudart, code)
                if text:
                    details.append(f"cudaGetDeviceCount_error={name} ({text})")
                else:
                    details.append(f"cudaGetDeviceCount_error={name}")
        except Exception as exc:
            details.append(f"cuda_probe_error={exc}")

        return details

    def _build_trt_runtime_error(self, error: Exception) -> RuntimeError:
        lines = [
            f"TensorRT Runtime 初始化失败: {error}",
            "这通常由 CUDA 驱动未就绪、驱动/运行时版本不匹配导致。",
            f"环境诊断: {'; '.join(self._collect_cuda_diagnostics())}",
            "可先执行 `python main.py verify-onnx` 验证 ONNX 推理链路。",
        ]
        return RuntimeError("\n".join(lines))

    def _load_engine(self, engine_path: Path):
        trt = self._get_trt()

        if not engine_path.exists():
            raise FileNotFoundError(engine_path)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        try:
            runtime = trt.Runtime(self._get_trt_logger())
        except Exception as exc:
            raise self._build_trt_runtime_error(exc) from exc

        if runtime is None:
            raise self._build_trt_runtime_error(
                RuntimeError("trt.Runtime returned None")
            )

        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        return engine

    @classmethod
    def _cache_key(cls, engine_path: Path) -> tuple[str, int, int]:
        resolved = engine_path.resolve()
        stat = resolved.stat()
        return (str(resolved), stat.st_mtime_ns, stat.st_size)

    def _get_cached_engine(self, engine_path: Path):
        cls = type(self)
        cache_key = self._cache_key(engine_path)

        with cls._engine_cache_lock:
            cached = cls._engine_cache.get(cache_key)
            if cached is not None:
                logger.info("复用已加载 TensorRT 引擎：%s", cache_key[0])
                return cached

            engine = self._load_engine(engine_path)
            cls._engine_cache = {
                key: value
                for key, value in cls._engine_cache.items()
                if key[0] != cache_key[0]
            }
            cls._engine_cache[cache_key] = engine
            logger.info("已缓存 TensorRT 引擎：%s", cache_key[0])
            return engine

    def _get_execution_context(self):
        if self.engine is None:
            raise RuntimeError("TensorRT engine 尚未加载")

        current_engine_id = id(self.engine)
        if self._context is not None and self._context_engine_id == current_engine_id:
            return self._context

        context = self.engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT create_execution_context failed")

        self._context = context
        self._context_engine_id = current_engine_id
        return context

    def infer_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trt = self._get_trt()
        self._ensure_cuda_ready()

        if self.engine is None:
            self.engine = self._get_cached_engine(self._resolve_engine_path())
            self._context = None
            self._context_engine_id = None

        from cuda.bindings import runtime as cudart

        def _cuda_check(result, op: str):
            if isinstance(result, tuple):
                code = result[0]
                payload = result[1:]
            else:
                code = result
                payload = ()

            if code != cudart.cudaError_t.cudaSuccess:
                name = str(code)
                detail = ""
                try:
                    err_name = cudart.cudaGetErrorName(code)
                    if isinstance(err_name, tuple) and len(err_name) > 1:
                        raw = err_name[1]
                        name = raw.decode() if isinstance(raw, bytes) else str(raw)

                    err_text = cudart.cudaGetErrorString(code)
                    if isinstance(err_text, tuple) and len(err_text) > 1:
                        raw = err_text[1]
                        detail = raw.decode() if isinstance(raw, bytes) else str(raw)
                except Exception:
                    pass

                msg = f"{op} failed: {name}"
                if detail:
                    msg = f"{msg} ({detail})"
                raise RuntimeError(msg)

            if not payload:
                return None
            if len(payload) == 1:
                return payload[0]
            return payload

        context = self._get_execution_context()

        input_hw = None
        if hasattr(self.engine, "num_io_tensors"):
            input_name = next(
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
                if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                == trt.TensorIOMode.INPUT
            )
            input_shape = tuple(self.engine.get_tensor_shape(input_name))
            if len(input_shape) >= 4 and input_shape[2] > 0 and input_shape[3] > 0:
                input_hw = (int(input_shape[2]), int(input_shape[3]))
        else:
            input_idx = next(
                i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)
            )
            input_shape = tuple(self.engine.get_binding_shape(input_idx))
            if len(input_shape) >= 4 and input_shape[2] > 0 and input_shape[3] > 0:
                input_hw = (int(input_shape[2]), int(input_shape[3]))

        tensor, working = self._preprocess(image, input_hw=input_hw)

        stream = None
        host = {}
        device = {}

        try:
            stream = _cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")
            if hasattr(self.engine, "num_io_tensors"):
                tensor_names = [
                    self.engine.get_tensor_name(i)
                    for i in range(self.engine.num_io_tensors)
                ]
                input_name = next(
                    name
                    for name in tensor_names
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                )
                output_name = next(
                    name
                    for name in tensor_names
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
                )

                context.set_input_shape(input_name, tensor.shape)

                for name in tensor_names:
                    shape = tuple(context.get_tensor_shape(name))
                    size = int(trt.volume(shape))
                    dtype = trt.nptype(self.engine.get_tensor_dtype(name))

                    host_buf = np.empty(size, dtype=dtype)
                    device_buf = _cuda_check(
                        cudart.cudaMalloc(host_buf.nbytes),
                        f"cudaMalloc(tensor={name})",
                    )

                    host[name] = host_buf
                    device[name] = int(device_buf)
                    context.set_tensor_address(name, int(device_buf))

                host[input_name] = np.ascontiguousarray(tensor.ravel())

                _cuda_check(
                    cudart.cudaMemcpyAsync(
                        device[input_name],
                        int(host[input_name].ctypes.data),
                        host[input_name].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    ),
                    "cudaMemcpyAsync(H2D)",
                )

                ok = context.execute_async_v3(stream_handle=int(stream))
                if not ok:
                    raise RuntimeError("TensorRT execute_async_v3 failed")

                _cuda_check(
                    cudart.cudaMemcpyAsync(
                        int(host[output_name].ctypes.data),
                        device[output_name],
                        host[output_name].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    "cudaMemcpyAsync(D2H)",
                )
            else:
                input_idx = output_idx = 0
                for i in range(self.engine.num_bindings):
                    if self.engine.binding_is_input(i):
                        input_idx = i
                    else:
                        output_idx = i

                context.set_binding_shape(input_idx, tensor.shape)
                bindings = [0] * self.engine.num_bindings

                for i in range(self.engine.num_bindings):
                    shape = tuple(context.get_binding_shape(i))
                    size = int(trt.volume(shape))
                    dtype = trt.nptype(self.engine.get_binding_dtype(i))

                    host_buf = np.empty(size, dtype=dtype)
                    device_buf = _cuda_check(
                        cudart.cudaMalloc(host_buf.nbytes),
                        f"cudaMalloc(binding={i})",
                    )

                    bindings[i] = int(device_buf)
                    host[i] = host_buf
                    device[i] = int(device_buf)

                host[input_idx] = np.ascontiguousarray(tensor.ravel())

                _cuda_check(
                    cudart.cudaMemcpyAsync(
                        device[input_idx],
                        int(host[input_idx].ctypes.data),
                        host[input_idx].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    ),
                    "cudaMemcpyAsync(H2D)",
                )

                ok = context.execute_async_v2(
                    bindings=bindings,
                    stream_handle=int(stream),
                )
                if not ok:
                    raise RuntimeError("TensorRT execute_async_v2 failed")

                _cuda_check(
                    cudart.cudaMemcpyAsync(
                        int(host[output_idx].ctypes.data),
                        device[output_idx],
                        host[output_idx].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    "cudaMemcpyAsync(D2H)",
                )

            _cuda_check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
        finally:
            for ptr in device.values():
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass
            if stream is not None:
                try:
                    cudart.cudaStreamDestroy(stream)
                except Exception:
                    pass

        if hasattr(self.engine, "num_io_tensors"):
            output_shape = tuple(context.get_tensor_shape(output_name))
            output = np.array(host[output_name]).reshape(output_shape)
        else:
            output_shape = tuple(context.get_binding_shape(output_idx))
            output = np.array(host[output_idx]).reshape(output_shape)

        if output.ndim == 3:
            output = np.expand_dims(output, axis=0)

        mask = self._logits_to_mask(output)
        overlay = self._build_overlay(working, mask)
        return mask, overlay

    def infer_file(self, path: Path) -> tuple[Path, Path]:
        image = self._load_image(path)
        mask, overlay = self.infer_image(image)
        return self._save_result(path, mask, overlay, "trt")

    # backward-compatible alias
    def infer_trt_file(self, path: Path) -> tuple[Path, Path]:
        return self.infer_file(path)

    def _resolve_engine_path(self) -> Path:
        return resolve_artifact_path(
            self.cfg.artifacts_dir,
            "model.trt",
            self.cfg.engine_path,
        )
