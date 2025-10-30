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


def main():
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN_THREE")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    config = Wan2_2_T2V_A14B_Config()

    # Combine HIGH and LOW LoRAs
    lora_high = "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-HIGH_converted.safetensors"
    lora_low = "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-LOW_converted.safetensors"

    # Weight ratio: 4 steps HIGH / 12 total = 0.33, 8 steps LOW / 12 total = 0.67
    combined_lora_path = "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-COMBINED_33high_67low.safetensors"

    if not Path(combined_lora_path).exists():
        print("\n" + "="*80)
        print("COMBINING HIGH + LOW LORAs")
        print("="*80)
        combine_loras(
            lora_paths=[lora_high, lora_low],
            weights=[0.33, 0.67],  # 4 steps / 12 total, 8 steps / 12 total
            output_path=combined_lora_path
        )
    else:
        print(f"\n✓ Combined LoRA already exists: {combined_lora_path}\n")

    # Initialize generator
    generator = VideoGenerator.from_pretrained(
        "/workspace/FastVideo/models/Wan2.2-T2V-A14B-Diffusers",
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        pipeline_config=config,
    )
    Path("/workspace/FastVideo/outputs").mkdir(parents=True, exist_ok=True)

    # Apply combined LoRA
    print("="*80)
    print("GENERATING WITH COMBINED HIGH+LOW LORA")
    print("="*80)
    generator.set_lora_adapter(
        lora_nickname="instagirl_combined",
        lora_path=combined_lora_path,
        lora_scale=1.1
    )

    prompt = (
        "Instagirl, casual mirror selfie with the iPhone 8 front camera "
        "(7MP, f/2.2, 1080p/30), walking in Central Park at late‑afternoon golden hour; "
        "head‑and‑shoulders at arm's length, mild wide‑angle feel, natural skin with fine pores, "
        "faint peach fuzz, tiny freckles and slight uneven tone (no beauty filter), lively eyes, "
        "subtle off‑white teeth with natural enamel translucency, a few hair flyaways, knit sweater "
        "and simple pendant; trees and a thin skyline band softly blurred behind, gentle sun flare, "
        "small hand‑held sway and a single natural blink, warm but true‑to‑life color, no oversharpening."
    )

    negative_prompt = "censored, sunburnt skin, rashy skin, red cheeks"

    output_path = "/workspace/FastVideo/outputs/test_combined_loras.mp4"
    generator.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1280,
        width=960,
        num_frames=150,
        num_inference_steps=12,
        fps=30,
        guidance_scale=1.0,
        save_video=True,
        output_path=output_path,
        seed=1024,
    )

    print("\n" + "="*80)
    print("✓ VIDEO GENERATED WITH COMBINED HIGH+LOW LORA")
    print("="*80)
    print(f"Output: {output_path}")
    print("\nThis combined LoRA should improve color vibrancy vs single HIGH or LOW LoRA.")
    print("Compare this to your previous videos to see if colors are more vibrant!")


if __name__ == "__main__":
    main()
