# SPDX-License-Identifier: MIT
"""
Lightweight RIFE v4.7 frame interpolation wrapper.

Adapted from https://github.com/Fannovel16/ComfyUI-Frame-Interpolation (MIT License).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["interpolate_rife_frames"]


# ---------------------------------------------------------------------------
#  Model building blocks (trimmed to the 4.7 architecture used by rife47.pth)
# ---------------------------------------------------------------------------

_BACKWARP_GRID: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}


def _default_device(device: torch.device | str | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _clear_cuda_if_needed(counter: int, threshold: int) -> int:
    if threshold <= 0:
        return counter
    if torch.cuda.is_available() and counter >= threshold:
        torch.cuda.empty_cache()
        return 0
    return counter


class ResConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)), requires_grad=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) * self.beta + x)


def warp(input_tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    2D warping utility used by the RIFE blocks.
    """
    device = input_tensor.device
    key = (str(device), tuple(flow.shape))
    if key not in _BACKWARP_GRID:
        horizontal = (
            torch.linspace(-1.0, 1.0, flow.shape[3], device=device)
            .view(1, 1, 1, flow.shape[3])
            .expand(flow.shape[0], -1, flow.shape[2], -1)
        )
        vertical = (
            torch.linspace(-1.0, 1.0, flow.shape[2], device=device)
            .view(1, 1, flow.shape[2], 1)
            .expand(flow.shape[0], -1, -1, flow.shape[3])
        )
        _BACKWARP_GRID[key] = torch.cat([horizontal, vertical], dim=1)

    flow = torch.cat(
        [
            flow[:, 0:1] / ((input_tensor.shape[3] - 1.0) / 2.0),
            flow[:, 1:2] / ((input_tensor.shape[2] - 1.0) / 2.0),
        ],
        dim=1,
    )
    grid = (_BACKWARP_GRID[key] + flow).permute(0, 2, 3, 1)
    if input_tensor.type() == "torch.cuda.HalfTensor":
        grid = grid.half()

    padding_mode = "border"
    if device.type == "mps":
        # https://github.com/pytorch/pytorch/issues/125098
        padding_mode = "zeros"
        grid = grid.clamp(-1, 1)

    return torch.nn.functional.grid_sample(
        input=input_tensor,
        grid=grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def _conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _deconv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True,
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )


class IFBlock(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            _conv(in_channels, base_channels // 2, stride=2),
            _conv(base_channels // 2, base_channels, stride=2),
        )
        self.body = nn.Sequential(
            ResConv(base_channels),
            ResConv(base_channels),
            ResConv(base_channels),
            ResConv(base_channels),
            ResConv(base_channels),
            ResConv(base_channels),
            ResConv(base_channels),
            ResConv(base_channels),
        )
        self.to_flow = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 4 * 6, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
                / scale
            )
            x = torch.cat((x, flow), dim=1)

        feat = self.body(self.down(x))
        tmp = self.to_flow(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow_out = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow_out, mask


class IFNet(nn.Module):
    """
    RIFE inference network specialised for v4.7 checkpoints.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ConvTranspose2d(16, 4, 4, 2, 1),
        )
        self.block0 = IFBlock(7 + 8, base_channels=192)
        self.block1 = IFBlock(8 + 4 + 8, base_channels=128)
        self.block2 = IFBlock(8 + 4 + 8, base_channels=96)
        self.block3 = IFBlock(8 + 4 + 8, base_channels=64)

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float | torch.Tensor = 0.5,
        scale_list: Sequence[float] | None = None,
    ) -> torch.Tensor:
        if scale_list is None:
            scale_list = (8.0, 4.0, 2.0, 1.0)

        img0 = torch.clamp(img0, 0, 1)
        img1 = torch.clamp(img1, 0, 1)

        n, _, h, w = img0.shape
        pad_h = ((h - 1) // 64 + 1) * 64
        pad_w = ((w - 1) // 64 + 1) * 64
        padding = (0, pad_w - w, 0, pad_h - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if not torch.is_tensor(timestep):
            timestep_tensor = img0.new_full((n, 1, img0.shape[2], img0.shape[3]), float(timestep))
        else:
            timestep_tensor = timestep.to(img0.device, img0.dtype)
            if timestep_tensor.ndim == 0:
                timestep_tensor = timestep_tensor.view(1, 1, 1, 1).expand(
                    n, 1, img0.shape[2], img0.shape[3]
                )
            elif timestep_tensor.ndim == 1:
                timestep_tensor = timestep_tensor.view(-1, 1, 1, 1).expand(
                    n, 1, img0.shape[2], img0.shape[3]
                )

        feat0 = self.encode(img0[:, :3])
        feat1 = self.encode(img1[:, :3])

        warped0 = img0
        warped1 = img1
        flow = None
        mask = None

        blocks = (self.block0, self.block1, self.block2, self.block3)
        for idx, block in enumerate(blocks):
            scale = scale_list[idx] if idx < len(scale_list) else scale_list[-1]
            if flow is None:
                flow, mask = block(
                    torch.cat((img0[:, :3], img1[:, :3], feat0, feat1, timestep_tensor), dim=1),
                    None,
                    scale=scale,
                )
            else:
                flow_delta, mask = block(
                    torch.cat(
                        (
                            warped0[:, :3],
                            warped1[:, :3],
                            warp(feat0, flow[:, :2]),
                            warp(feat1, flow[:, 2:4]),
                            timestep_tensor,
                            mask,
                        ),
                        dim=1,
                    ),
                    flow,
                    scale=scale,
                )
                flow = flow + flow_delta

            warped0 = warp(img0, flow[:, :2])
            warped1 = warp(img1, flow[:, 2:4])

        assert mask is not None
        mask = torch.sigmoid(mask)
        merged = warped0 * mask + warped1 * (1.0 - mask)
        return torch.clamp(merged[:, :, :h, :w], 0.0, 1.0)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        raise ValueError("Expected frames shaped [H, W, 3]")
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=torch.float32)


def _tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return (array * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def _load_rife_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    # PyTorch 2.6 flipped torch.load(weights_only) default to True, which breaks
    # legacy checkpoints that store optimizer metadata. Force False explicitly.
    state = torch.load(
        ckpt_path.as_posix(),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def interpolate_rife_frames(
    frames: Sequence[np.ndarray],
    ckpt_path: Path | str,
    multiplier: int = 2,
    clear_cache_after_n_frames: int = 6,
    device: torch.device | str | None = None,
) -> list[np.ndarray]:
    """
    Interpolate intermediate frames using a RIFE v4.7 checkpoint.

    Args:
        frames: Sequence of HxWx3 uint8 frames to interpolate.
        ckpt_path: Path to a rife4.7-compatible .pth checkpoint.
        multiplier: Number of segments per interval (2 -> insert one frame).
        clear_cache_after_n_frames: How frequently to empty CUDA cache.
        device: Specific torch device to run on (defaults to CUDA if available).
    """
    if multiplier <= 1 or len(frames) < 2:
        return list(frames)

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"RIFE checkpoint not found: {ckpt}")

    runtime_device = _default_device(device)
    model = IFNet()
    model.load_state_dict(_load_rife_state_dict(ckpt))
    model.to(runtime_device)
    model.eval()

    output_frames: list[np.ndarray] = []
    cache_counter = 0

    context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with context():
        prev_tensor = _frame_to_tensor(frames[0], runtime_device)
        output_frames.append(np.array(frames[0], copy=True))

        for idx in range(1, len(frames)):
            next_tensor = _frame_to_tensor(frames[idx], runtime_device)

            for step in range(1, multiplier):
                timestep = step / float(multiplier)
                middle = model(prev_tensor, next_tensor, timestep=timestep)
                output_frames.append(_tensor_to_frame(middle))
                cache_counter += 1
                cache_counter = _clear_cuda_if_needed(cache_counter, clear_cache_after_n_frames)

            output_frames.append(np.array(frames[idx], copy=True))
            prev_tensor = next_tensor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_frames
