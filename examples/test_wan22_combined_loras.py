"""
Combined HIGH+LOW LoRA test to approximate two-stage effect.

Strategy: Combine HIGH (weight 0.33) + LOW (weight 0.67) LoRAs
- Mimics 4 steps HIGH + 8 steps LOW in 12 total steps
- Should improve color vibrancy vs single LoRA
"""
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import Wan2_2_T2V_A14B_Config

MODEL_PATH = "/workspace/FastVideo/models/Wan2.2-T2V-A14B-Diffusers"
OUTPUT_DIR = Path("/workspace/FastVideo/outputs")
LORA_HIGH = Path(
    "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-HIGH_converted.safetensors")
LORA_LOW = Path(
    "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-LOW_converted.safetensors")
COMBINED_LORA = Path(
    "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-COMBINED_33high_67low.safetensors")

_GENERATOR: VideoGenerator | None = None
_CURRENT_LORA: tuple[str, float] | None = None


def combine_loras(lora_paths, weights, output_path):
    """
    Combine multiple LoRAs with given weights.

    Args:
        lora_paths: List of paths to LoRA files
        weights: List of weights for each LoRA (should sum to ~1.0)
        output_path: Where to save combined LoRA
    """
    print(f"\nCombining {len(lora_paths)} LoRAs:")
    for path, weight in zip(lora_paths, weights):
        print(f"  {Path(path).name}: weight={weight}")

    state_dicts = [load_file(str(path)) for path in lora_paths]

    combined_state = {}
    all_modules = set()

    # Discover all LoRA modules (FastVideo converted format uses lora_A/lora_B)
    for state in state_dicts:
        modules = [
            key.replace(".lora_A", "")
            for key in state.keys()
            if key.endswith("lora_A")
        ]
        all_modules.update(modules)

    all_modules = sorted(all_modules)
    print(f"  Processing {len(all_modules)} LoRA modules...")

    for module in all_modules:
        up_chunks = []
        down_chunks = []
        total_rank = 0
        dtype = None

        for state, weight in zip(state_dicts, weights):
            lora_a_key = f"{module}.lora_A"
            lora_b_key = f"{module}.lora_B"
            alpha_key = f"{module}.alpha"

            if lora_a_key not in state or lora_b_key not in state:
                continue

            lora_a = state[lora_a_key].to(torch.float32)
            lora_b = state[lora_b_key].to(torch.float32)
            rank = lora_a.shape[0]
            dtype = state[lora_a_key].dtype

            alpha = state.get(alpha_key)
            base_alpha = alpha.item() if alpha is not None else rank
            scale = weight * (float(base_alpha) / rank)

            lora_b_scaled = lora_b * scale

            up_chunks.append(lora_b_scaled)
            down_chunks.append(lora_a)
            total_rank += rank

        if not up_chunks:
            continue

        if dtype is None:
            dtype = torch.float16

        # Concatenate along rank dimension
        lora_b_new = torch.cat(up_chunks, dim=1).to(dtype)
        lora_a_new = torch.cat(down_chunks, dim=0).to(dtype)

        combined_state[f"{module}.lora_B"] = lora_b_new
        combined_state[f"{module}.lora_A"] = lora_a_new
        combined_state[f"{module}.alpha"] = torch.tensor(float(total_rank), dtype=dtype)

    save_file(combined_state, str(output_path))
    print(f"  ✓ Saved combined LoRA: {output_path}")
    print(f"    {len(combined_state)//3} modules, total size: {Path(output_path).stat().st_size / 1024**2:.1f} MB\n")
    return str(output_path)


def get_video_generator() -> VideoGenerator:
    """Lazy-load and cache a VideoGenerator instance for reuse."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    config = Wan2_2_T2V_A14B_Config()

    _GENERATOR = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        pipeline_config=config,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _GENERATOR


def shutdown_generator() -> None:
    """Cleanly tear down the cached generator (if any)."""
    global _GENERATOR, _CURRENT_LORA
    if _GENERATOR is not None:
        _GENERATOR.shutdown()
    _GENERATOR = None
    _CURRENT_LORA = None


def prepare_combined_lora() -> Path:
    """Ensure the combined LoRA exists on disk and return its path."""
    # Weight ratio: 4 steps HIGH / 12 total = 0.33, 8 steps LOW / 12 total = 0.67
    if not COMBINED_LORA.exists():
        print("\n" + "="*80)
        print("COMBINING HIGH + LOW LORAs")
        print("="*80)
        combine_loras(
            lora_paths=[LORA_HIGH, LORA_LOW],
            weights=[0.33, 0.67],  # 4 steps / 12 total, 8 steps / 12 total
            output_path=COMBINED_LORA
        )
    else:
        print(f"\n✓ Combined LoRA already exists: {COMBINED_LORA}\n")
    return COMBINED_LORA


def load_combined_lora(lora_scale: float = 1.1) -> VideoGenerator:
    """
    Load (or reuse) the cached generator and apply the combined LoRA.

    LoRA application is idempotent—if the same adapter and scale were already
    applied, we skip the call to avoid redundant synchronization/barriers.
    """
    global _CURRENT_LORA
    generator = get_video_generator()
    combined_lora_path = prepare_combined_lora()

    cache_key = (combined_lora_path.as_posix(), lora_scale)
    if _CURRENT_LORA != cache_key:
        print("="*80)
        print("GENERATING WITH COMBINED HIGH+LOW LORA")
        print("="*80)
        generator.set_lora_adapter(
            lora_nickname="instagirl_combined",
            lora_path=combined_lora_path.as_posix(),
            lora_scale=lora_scale,
        )
        _CURRENT_LORA = cache_key
    else:
        print("="*80)
        print("REUSING CACHED GENERATOR + COMBINED LORA")
        print("="*80)
    return generator


def main():
    generator = load_combined_lora(lora_scale=1.1)

    prompt = ""

    negative_prompt = ""

    output_path = OUTPUT_DIR / "test_combined_loras.mp4"
    generator.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1280,
        width=960,
        num_frames=75,
        num_inference_steps=12,
        fps=15,
        guidance_scale=4.0,
        save_video=True,
        output_path=output_path.as_posix(),
        seed=1024,
    )

    print("\n" + "="*80)
    print("✓ VIDEO GENERATED WITH COMBINED HIGH+LOW LORA")
    print("="*80)
    print(f"Output: {output_path.as_posix()}")
    print("\nThis combined LoRA should improve color vibrancy vs single HIGH or LOW LoRA.")
    print("Compare this to your previous videos to see if colors are more vibrant!")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_generator()
