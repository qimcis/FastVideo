"""
Dual-LoRA inference for Wan2.2 using dedicated high/low adapters plus a helper LoRA.

Setup:
- High-noise expert gets Instagirl HIGH LoRA blended with lightx2v helper.
- Low-noise expert gets Instagirl LOW LoRA blended with the same helper.
- Runtime uses VideoGenerator.set_dual_lora_adapters so the experts stay separated.

The helper blend mimics the ComfyUI Instagirl workflow (helper strength 0.6,
primary Instagirl LoRAs at full strength), while the sampler runs at CFG 4.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import Wan2_2_T2V_A14B_Config


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

_GENERATOR: VideoGenerator | None = None


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


def get_video_generator() -> VideoGenerator:
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    config = Wan2_2_T2V_A14B_Config()
    _GENERATOR = VideoGenerator.from_pretrained(
        MODEL_PATH.as_posix(),
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        pipeline_config=config,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _GENERATOR


def main() -> None:
    generator = get_video_generator()
    lora_high_aug, lora_low_aug = prepare_augmented_loras()

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
        ""
    )

    negative_prompt = (
        "perfectly smooth skin, glass skin"
    )


    output_path = OUTPUT_DIR / "test_dual_lora_with_helper.mp4"
    generator.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1280,
        width=960,
        num_frames=75,
        num_inference_steps=NUM_STEPS,
        fps=15,
        guidance_scale=CFG_SCALE,
        boundary_ratio=BOUNDARY_RATIO,
        save_video=True,
        output_path=output_path.as_posix(),
        seed=1024,
    )

    print("\n" + "=" * 80)
    print("✓ VIDEO GENERATED WITH DUAL LORAs + LIGHTX2V HELPER")
    print("=" * 80)
    print(f"Output: {output_path.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _GENERATOR is not None:
            _GENERATOR.shutdown()
            _GENERATOR = None
