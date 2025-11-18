"""
Minimal imageio stub for offline environments.

Only implements ``mimsave`` using ffmpeg so FastVideo scripts can write MP4
videos without the full imageio dependency.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def mimsave(
    output_path: str | os.PathLike[str],
    frames: Sequence[np.ndarray] | Iterable[np.ndarray],
    *,
    fps: int = 24,
    format: str = "mp4",
) -> None:
    """
    Save a list of RGB frames to disk using ffmpeg.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("No frames provided to mimsave()")

    height, width, channels = frame_list[0].shape
    if channels != 3:
        raise ValueError("Expected RGB frames shaped [H, W, 3]")

    for idx, frame in enumerate(frame_list):
        if frame.shape != frame_list[0].shape:
            raise ValueError(
                f"Frame {idx} has mismatched shape {frame.shape}, expected {frame_list[0].shape}"
            )

    output_path = Path(output_path).as_posix()
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frame_list:
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with status {ret}")


def get_reader(*_args, **_kwargs):  # pragma: no cover - not needed for current scripts
    raise NotImplementedError("imageio.get_reader is not implemented in the stub module")

