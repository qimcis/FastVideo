"""
Dual-LoRA inference for Wan2.2 using dedicated high/low adapters plus a helper LoRA.

Setup:
- High-noise expert gets Instagirl HIGH LoRA blended with lightx2v helper.
- Low-noise expert gets Instagirl LOW LoRA blended with the same helper.
- Runtime uses VideoGenerator.set_dual_lora_adapters so the experts stay separated.

The helper blend mimics the ComfyUI Instagirl workflow (helper strength 0.6,
primary Instagirl LoRAs at full strength), while the sampler runs at CFG 4.

This script now renders an 8 fps, 5 second base clip (40 frames). The final
stage can optionally:
- Run Real-ESRGAN super-resolution on every frame (default in this revision).
- Run RIFE interpolation (disabled by default) to reach higher frame rates.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import imageio
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import Wan2_2_T2V_A14B_Config
from fastvideo.configs.sample.wan import WanT2V_14B_SamplingParam
from fastvideo.interpolation import interpolate_rife_frames


MODEL_PATH = Path("/workspace/FastVideo/models/Wan2.2-T2V-A14B-Diffusers")
OUTPUT_DIR = Path("/workspace/FastVideo/outputs")

LORA_ROOT = Path("/workspace/FastVideo/loras")
INSTAGIRL_ROOT = LORA_ROOT / "Instagirlv2.5"

LORA_HIGH = INSTAGIRL_ROOT / "Instagirlv2.5-HIGH_converted.safetensors"
LORA_LOW = INSTAGIRL_ROOT / "Instagirlv2.5-LOW_converted.safetensors"

# Helper LoRAs converted to FastVideo format (one per noise expert).
LORA_HELPER_HIGH = (
    LORA_ROOT
    / "lightx2v"
    / "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022_converted.safetensors"
)
LORA_HELPER_LOW = (
    LORA_ROOT
    / "lightx2v"
    / "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022_converted.safetensors"
)

LORA_HIGH_PLUS_HELPER = (
    INSTAGIRL_ROOT / "Instagirlv2.5-HIGH_plus_lightx2v60.safetensors"
)
LORA_LOW_PLUS_HELPER = (
    INSTAGIRL_ROOT / "Instagirlv2.5-LOW_plus_lightx2v60.safetensors"
)

INSTAGIRL_SCALE = 1.0
HELPER_SCALE = 0.6
DUAL_LORA_SCALE = 1.1  

CFG_SCALE = 4.0
BOUNDARY_RATIO = 0.965
NUM_STEPS = 12
TEACACHE_THRESHOLD = 0.12

# Video timing parameters
CLIP_DURATION_SECONDS = 5
BASE_FPS = 8
TARGET_FPS = 24
USE_FRAME_INTERPOLATION = False  # disable when leaning on super-resolution instead of RIFE
SAVE_BASE_CLIP = True
RIFE_CHECKPOINT = Path("/workspace/FastVideo/models/rife/rife47.pth")
INTERPOLATION_CLEAR_CACHE = 6

# Super-resolution parameters (Real-ESRGAN x4)
USE_SUPER_RESOLUTION = True
# Download the original PyTorch checkpoint from https://ai-modelscope.oss-cn-beijing.aliyuncs.com/openmmlab/realesrgan/RealESRGAN_x4plus.pth
REALESRGAN_WEIGHTS = Path("/workspace/FastVideo/models/realesrgan/RealESRGAN_x4plus.pth")
SUPERRES_SCALE = 2.0  # 4.0 is available, but 2.0 keeps VRAM and processing time manageable
SUPERRES_TILE = 0  # 0 = auto; set to 512 if you need tiling on lower-memory GPUs
SUPERRES_TILE_PAD = 10
SUPERRES_PREPAD = 0

if TARGET_FPS % BASE_FPS != 0:
    raise ValueError(
        f"TARGET_FPS ({TARGET_FPS}) must be a multiple of BASE_FPS ({BASE_FPS}) "
        "for frame interpolation."
    )

INTERPOLATION_MULTIPLIER = TARGET_FPS // BASE_FPS
BASE_NUM_FRAMES = int(CLIP_DURATION_SECONDS * BASE_FPS)
BASE_OUTPUT_NAME = f"test_dual_lora_with_helper_base_{BASE_FPS}fps.mp4"
FINAL_OUTPUT_NAME = (
    f"test_dual_lora_with_helper_{TARGET_FPS if USE_FRAME_INTERPOLATION else BASE_FPS}fps.mp4"
)

_GENERATOR: VideoGenerator | None = None
_REALESRGANER: Any | None = None


def _check_files_exist(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        formatted = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Required LoRA file(s) not found. Please convert them to FastVideo format first:\n"
            f"  {formatted}"
        )


def blend_loras(lora_paths: list[Path], weights: list[float], output_path: Path) -> Path:
    """
    Concatenate multiple LoRAs (keeping ranks separate) so their deltas add linearly.

    Args:
        lora_paths: Paths to source LoRAs in FastVideo format (lora_A/lora_B).
        weights: Per-LoRA blend weights (approximate relative strength).
        output_path: Destination file for the blended LoRA.
    """
    if len(lora_paths) != len(weights):
        raise ValueError("lora_paths and weights must have the same length")
    print(f"\nBlending {len(lora_paths)} LoRAs into {output_path.name}")
    for src, weight in zip(lora_paths, weights):
        print(f"  - {src.name}: weight={weight}")

    state_dicts = [load_file(path.as_posix()) for path in lora_paths]
    all_modules = sorted({
        key.removesuffix(".lora_A")
        for state in state_dicts
        for key in state.keys()
        if key.endswith("lora_A")
    })

    blended_state: dict[str, torch.Tensor] = {}
    for module in all_modules:
        accum_a: list[torch.Tensor] = []
        accum_b: list[torch.Tensor] = []
        total_rank = 0
        dtype: torch.dtype | None = None

        for state, weight in zip(state_dicts, weights):
            key_a = f"{module}.lora_A"
            key_b = f"{module}.lora_B"
            key_alpha = f"{module}.alpha"

            if key_a not in state or key_b not in state:
                continue

            lora_a = state[key_a].to(torch.float32)
            lora_b = state[key_b].to(torch.float32)
            rank = lora_a.shape[0]
            dtype = state[key_a].dtype

            alpha_tensor = state.get(key_alpha)
            base_alpha = alpha_tensor.item() if alpha_tensor is not None else rank
            scale = weight * (float(base_alpha) / rank)

            accum_a.append(lora_a)
            accum_b.append(lora_b * scale)
            total_rank += rank

        if not accum_a:
            continue

        assert dtype is not None
        blended_state[f"{module}.lora_A"] = torch.cat(accum_a, dim=0).to(dtype)
        blended_state[f"{module}.lora_B"] = torch.cat(accum_b, dim=1).to(dtype)
        blended_state[f"{module}.alpha"] = torch.tensor(float(total_rank), dtype=dtype)

    save_file(blended_state, output_path.as_posix())
    print(f"  ✓ Saved {output_path.name} ({len(blended_state) // 3} modules)")
    return output_path


def prepare_augmented_loras() -> tuple[Path, Path]:
    """Ensure helper-augmented LoRAs exist for both experts."""
    _check_files_exist([LORA_HIGH, LORA_LOW, LORA_HELPER_HIGH, LORA_HELPER_LOW])

    if not LORA_HIGH_PLUS_HELPER.exists():
        blend_loras(
            [LORA_HIGH, LORA_HELPER_HIGH],
            [INSTAGIRL_SCALE, HELPER_SCALE],
            LORA_HIGH_PLUS_HELPER,
        )
    else:
        print(f"\n✓ Reusing cached helper blend: {LORA_HIGH_PLUS_HELPER}")

    if not LORA_LOW_PLUS_HELPER.exists():
        blend_loras(
            [LORA_LOW, LORA_HELPER_LOW],
            [INSTAGIRL_SCALE, HELPER_SCALE],
            LORA_LOW_PLUS_HELPER,
        )
    else:
        print(f"\n✓ Reusing cached helper blend: {LORA_LOW_PLUS_HELPER}")

    return LORA_HIGH_PLUS_HELPER, LORA_LOW_PLUS_HELPER


def _require_file(path: Path, description: str, *, min_size_bytes: int = 1_000_000) -> Path:
    """
    Ensure a model checkpoint exists and is not trivially corrupted.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")

    size = path.stat().st_size
    if size < min_size_bytes:
        raise FileNotFoundError(
            f"{description} appears corrupted or incomplete: {path} "
            f"(only {size} bytes). Re-download it before running again."
        )
    return path


def _require_rife_checkpoint(path: Path) -> Path:
    """
    Ensure the configured RIFE checkpoint exists.
    """
    return _require_file(
        path,
        "RIFE checkpoint (rife47.pth)",
        min_size_bytes=50_000_000,
    )


def _get_realesrgan() -> Any:
    """
    Lazily instantiate a Real-ESRGAN upsampler.
    """
    global _REALESRGANER
    if _REALESRGANER is not None:
        return _REALESRGANER

    try:
        import torchvision  # noqa: F401
        from torchvision.transforms import functional as tv_functional
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for Real-ESRGAN. Install with:\n"
            "  pip install torchvision\n"
        ) from exc

    # TorchVision 0.22 removed torchvision.transforms.functional_tensor; stub it if needed.
    try:
        import torchvision.transforms.functional_tensor  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("torchvision.transforms.functional_tensor")

        def _rgb_to_grayscale(img, num_output_channels: int = 1):
            return tv_functional.rgb_to_grayscale(img, num_output_channels=num_output_channels)

        module.rgb_to_grayscale = _rgb_to_grayscale  # type: ignore[attr-defined]
        sys.modules["torchvision.transforms.functional_tensor"] = module

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Real-ESRGAN is not installed. Install dependencies with:\n"
            "  pip install realesrgan basicsr facexlib\n"
            "Then download RealESRGAN_x4plus.pth into "
            f"{REALESRGAN_WEIGHTS.parent}"
        ) from exc

    _require_file(REALESRGAN_WEIGHTS, "Real-ESRGAN x4 weights", min_size_bytes=50_000_000)

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    _REALESRGANER = RealESRGANer(
        model_path=REALESRGAN_WEIGHTS.as_posix(),
        model=model,
        scale=4,
        tile=SUPERRES_TILE,
        tile_pad=SUPERRES_TILE_PAD,
        pre_pad=SUPERRES_PREPAD,
        half=torch.cuda.is_available(),
    )
    return _REALESRGANER


def upscale_frames_with_realesrgan(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    """
    Apply Real-ESRGAN super-resolution to every frame.
    """
    upsampler = _get_realesrgan()
    upscaled: list[np.ndarray] = []
    total = len(frames)
    for idx, frame in enumerate(frames, start=1):
        bgr_frame = frame[:, :, ::-1].copy()
        sr_bgr, _ = upsampler.enhance(
            bgr_frame,
            outscale=SUPERRES_SCALE,
        )
        sr_rgb = sr_bgr[:, :, ::-1]
        upscaled.append(sr_rgb.astype(np.uint8))
        if idx % 10 == 0 or idx == total:
            print(f"  - Upscaled {idx}/{total} frames")
    return upscaled


def get_video_generator() -> VideoGenerator:
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    # Flash Attention v2/v3 is consistently faster (and artifact-free) on dual H200s.
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    config = Wan2_2_T2V_A14B_Config()
    _GENERATOR = VideoGenerator.from_pretrained(
        MODEL_PATH.as_posix(),
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        pipeline_config=config,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        pin_cpu_memory=False,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _GENERATOR


def main() -> None:
    generator = get_video_generator()
    lora_high_aug, lora_low_aug = prepare_augmented_loras()
    sampling_params = WanT2V_14B_SamplingParam()
    sampling_params.teacache_params.teacache_thresh = TEACACHE_THRESHOLD

    print("\n" + "=" * 80)
    print("LOADING DUAL LORAs (HIGH/LOW) WITH LIGHTX2V HELPER")
    print("=" * 80)
    generator.set_dual_lora_adapters(
        lora_high_nickname="instagirl_high_helper",
        lora_high_path=lora_high_aug.as_posix(),
        lora_low_nickname="instagirl_low_helper",
        lora_low_path=lora_low_aug.as_posix(),
        lora_scale=DUAL_LORA_SCALE,
    )

    prompt = (
        "mirror selfie, 20-year-old Korean beauty, long straight jet-black hair, perfect symmetrical face, "
        "big doe eyes, long lashes, small nose, plump lips with natural gloss, slight smile, "
        "wearing tight black long-sleeve crop top with deep V-neck, matching black high-waist micro mini skirt, "
        "natural hourglass figure, manicured white nail, iPhone 15 Pro Max in black case held at eye level, "
        "ornate white door frame behind, soft bathroom lighting, ultra-realistic skin texture with micro-pores "
        "and peach fuzz and natural blemishes, "
        "85mm f/1.2 prime lens, EXTREME shallow depth of field, eyes + phone screen tack-sharp, "
        "background door dissolves into huge pastel bokeh orbs "
        "shot on iPhone, cinematic color grade, 8K, hyper-detailed"
    )
    
    negative_prompt = (
        "blurry, low-res, pixelated, deformed hands, extra fingers, missing fingers, fused fingers, "
        "bad anatomy, watermark, text, logo, username, out of frame, cropped head, plastic skin, doll face, "
        "overexposed, underexposed, cross-eyed, makeup smudge, clothing wrinkles, motion blur, "
        "3d render, cartoon, anime, painting, sketch, extra limbs, mutated, disfigured, low-poly, "
        "artifacts, jpeg noise, deep depth of field, f/5.6, f/8, sharp background, no bokeh"
    )


    base_output_path = OUTPUT_DIR / BASE_OUTPUT_NAME
    final_output_path = OUTPUT_DIR / FINAL_OUTPUT_NAME

    result = generator.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        sampling_param=sampling_params,
        height=1280,
        width=960,
        num_frames=BASE_NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        fps=BASE_FPS,
        guidance_scale=CFG_SCALE,
        boundary_ratio=BOUNDARY_RATIO,
        save_video=False,
        return_frames=True,
        output_path=base_output_path.as_posix(),
        seed=1024,
        enable_teacache=True,
    )

    frames: list[np.ndarray]
    if isinstance(result, dict):
        frames = result["frames"]
    else:
        frames = result
    if SAVE_BASE_CLIP:
        print(f"\nSaving base {BASE_FPS}fps clip to {base_output_path}")
        imageio.mimsave(base_output_path.as_posix(), frames, fps=BASE_FPS, format="mp4")

    final_frames = frames
    final_fps = BASE_FPS
    if USE_FRAME_INTERPOLATION:
        ckpt_path = _require_rife_checkpoint(RIFE_CHECKPOINT)
        print(f"\nRunning RIFE interpolation ({ckpt_path.name}, x{INTERPOLATION_MULTIPLIER})...")
        final_frames = interpolate_rife_frames(
            frames,
            ckpt_path=ckpt_path,
            multiplier=INTERPOLATION_MULTIPLIER,
            clear_cache_after_n_frames=INTERPOLATION_CLEAR_CACHE,
        )
        final_fps = TARGET_FPS
    if USE_SUPER_RESOLUTION:
        print("\nRunning Real-ESRGAN super-resolution...")
        final_frames = upscale_frames_with_realesrgan(final_frames)

    print(f"\nSaving final {final_fps}fps clip to {final_output_path}")
    imageio.mimsave(final_output_path.as_posix(), final_frames, fps=final_fps, format="mp4")

    print("\n" + "=" * 80)
    print("✓ VIDEO GENERATED WITH DUAL LORAs + LIGHTX2V HELPER")
    if USE_FRAME_INTERPOLATION:
        print(f"  (RIFE interpolated from {BASE_FPS}fps to {TARGET_FPS}fps)")
    if USE_SUPER_RESOLUTION:
        print(f"  (Real-ESRGAN upscaled x{SUPERRES_SCALE})")
    print("=" * 80)
    print(f"Output: {final_output_path.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _GENERATOR is not None:
            _GENERATOR.shutdown()
            _GENERATOR = None
