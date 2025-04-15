# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/cuda.py
"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from functools import lru_cache, wraps
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import torch
# NOTE(will): this import is necessary to trigger the registration of the custom
# ops from vllm, which we use
# import custom ops, trigger op registration
import vllm._C  # noqa
from typing_extensions import ParamSpec

import fastvideo.v1.envs as envs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.platforms.interface import (DeviceCapability, Platform,
                                              PlatformEnum, _Backend)
from fastvideo.v1.utils import import_pynvml

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

pynvml = import_pynvml()  # type: ignore[no-untyped-call]

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
torch.backends.cuda.enable_cudnn_sdp(False)


def device_id_to_physical_device_id(device_id: int) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            msg = (
                "CUDA_VISIBLE_DEVICES is set to empty string, which means"
                " GPU support is disabled. If you are using ray, please unset"
                " the environment variable `CUDA_VISIBLE_DEVICES` inside the"
                " worker/actor. "
                "Check https://github.com/vllm-project/vllm/issues/8402 for"
                " more information.")
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


class CudaPlatformBase(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls,
                              device_id: int = 0) -> Optional[DeviceCapability]:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used")
            return False
        return True

    @classmethod
    def is_full_nvlink(cls, device_ids: List[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls) -> None:
        pass

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: Optional[_Backend],
                             head_size: int, dtype: torch.dtype) -> str:
        # TODO(will): maybe come up with a more general interface for local attention
        # if distributed is False, we always try to use Flash attn

        logger.info("Trying FASTVIDEO_ATTENTION_BACKEND=%s",
                    envs.FASTVIDEO_ATTENTION_BACKEND)
        if selected_backend == _Backend.SLIDING_TILE_ATTN:
            try:
                from st_attn import sliding_tile_attention  # noqa: F401

                from fastvideo.v1.attention.backends.sliding_tile_attn import (  # noqa: F401
                    SlidingTileAttentionBackend)
                logger.info("Using Sliding Tile Attention backend.")
                return "fastvideo.v1.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend"
            except ImportError as e:
                # TODO(will): improve error message
                logger.info(e)
                logger.info(
                    "Sliding Tile Attention backend is not installed. Fall back to Flash Attention."
                )
        elif selected_backend == _Backend.TORCH_SDPA:
            return "fastvideo.v1.attention.backends.sdpa.SDPABackend"
        elif selected_backend == _Backend.FLASH_ATTN or selected_backend is None:
            pass
        elif selected_backend:
            raise ValueError(f"Invalid attention backend for {cls.device_name}")

        target_backend = _Backend.FLASH_ATTN
        if not cls.has_device_capability(80):
            logger.info(
                "Cannot use FlashAttention-2 backend for Volta and Turing "
                "GPUs.")
            target_backend = _Backend.TORCH_SDPA
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention-2 backend for dtype other than "
                "torch.float16 or torch.bfloat16.")
            target_backend = _Backend.TORCH_SDPA

        # FlashAttn is valid for the model, checking if the package is
        # installed.
        if target_backend == _Backend.FLASH_ATTN:
            try:
                import flash_attn  # noqa: F401

                from fastvideo.v1.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend)

                supported_sizes = \
                    FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention-2 backend for head size %d.",
                        head_size)
                    target_backend = _Backend.TORCH_SDPA
            except ImportError:
                logger.info("Cannot use FlashAttention-2 backend because the "
                            "flash_attn package is not found. "
                            "Make sure that flash_attn was built and installed "
                            "(on by default).")
                target_backend = _Backend.TORCH_SDPA

        if target_backend == _Backend.TORCH_SDPA:
            logger.info(
                "Using torch.nn.functional.scaled_dot_product_attention backend."
            )
            return "fastvideo.v1.attention.backends.sdpa.SDPABackend"

        logger.info("Using Flash Attention backend.")
        return "fastvideo.v1.attention.backends.flash_attn.FlashAttentionBackend"

        target_backend = _Backend.FLASH_ATTN
        if not cls.has_device_capability(80):
            logger.info(
                "Cannot use FlashAttention-2 backend for Volta and Turing "
                "GPUs.")
            target_backend = _Backend.TORCH_SDPA
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention-2 backend for dtype=%s (other than torch.float16 or torch.bfloat16).",
                dtype)
            target_backend = _Backend.TORCH_SDPA

        # FlashAttn is valid for the model, checking if the package is
        # installed.
        if target_backend == _Backend.FLASH_ATTN:
            try:
                import flash_attn  # noqa: F401

                from fastvideo.v1.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend)

                supported_sizes = \
                    FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention-2 backend for head size %d.",
                        head_size)
                    target_backend = _Backend.TORCH_SDPA
            except ImportError:
                logger.info("Cannot use FlashAttention-2 backend because the "
                            "flash_attn package is not found. "
                            "Make sure that flash_attn was built and installed "
                            "(on by default).")
                target_backend = _Backend.TORCH_SDPA

        if target_backend == _Backend.TORCH_SDPA:
            # TODO(will): Implement torch SDPA backend.
            raise NotImplementedError("Torch SDPA is not implemented yet.")
            logger.info(
                "Using torch.nn.functional.scaled_dot_product_attention backend."
            )
            return "fastvideo.v1.attention.backends.torch_sdpa.TorchSDPA"

        logger.info("Using Flash Attention backend.")
        return "fastvideo.v1.attention.backends.flash_attn.FlashAttentionBackend"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "fastvideo.v1.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class NvmlCudaPlatform(CudaPlatformBase):

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_capability(cls,
                              device_id: int = 0) -> Optional[DeviceCapability]:
        try:
            physical_device_id = device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def has_device_capability(
        cls,
        capability: Union[Tuple[int, int], int],
        device_id: int = 0,
    ) -> bool:
        try:
            return bool(super().has_device_capability(capability, device_id))
        except RuntimeError:
            return False

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return str(pynvml.nvmlDeviceGetUUID(handle))

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def is_full_nvlink(cls, physical_device_ids: List[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped.")
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return str(pynvml.nvmlDeviceGetName(handle))

    @classmethod
    @with_nvml_context
    def log_warnings(cls) -> None:
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [
                cls._get_physical_device_name(i) for i in range(device_ids)
            ]
            if (len(set(device_names)) > 1
                    and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonNvmlCudaPlatform(CudaPlatformBase):

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return int(device_props.total_memory)

    @classmethod
    def is_full_nvlink(cls, physical_device_ids: List[int]) -> bool:
        logger.exception("NVLink detection not possible, as context support was"
                         " not found. Assuming no NVLink available.")
        return False


# Autodetect either NVML-enabled or non-NVML platform
# based on whether NVML is available.
nvml_available = False
try:
    try:
        pynvml.nvmlInit()
        nvml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        nvml_available = False
finally:
    if nvml_available:
        pynvml.nvmlShutdown()

CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform

try:
    from sphinx.ext.autodoc.mock import _MockModule

    if not isinstance(pynvml, _MockModule):
        CudaPlatform.log_warnings()
except ModuleNotFoundError:
    CudaPlatform.log_warnings()
