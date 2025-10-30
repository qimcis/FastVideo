"""
Test script for Wan2.2-T2V-A14B with LoRA support.

Note: This script includes a combine_loras() function for merging multiple LoRAs,
but currently demonstrates single LoRA usage.

LoRA Format: Official Wan training produces LoRAs with 'lora_unet_' prefix.
These need to be converted to FastVideo's expected format using convert_wan_lora.py:
  python3 convert_wan_lora.py path/to/original.safetensors path/to/converted.safetensors
"""
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import Wan2_2_T2V_A14B_Config


def combine_loras(lora_dirs, weights, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_files = [Path(d) / "pytorch_lora_weights.safetensors" for d in lora_dirs]

    print(f"Combining {len(lora_files)} LoRAs -> {output_dir}")
    state_dicts = []
    for idx, path in enumerate(lora_files):
        print(f"  Loading LoRA {idx}: {path}")
        state_dicts.append(load_file(str(path)))

    combined_state = {}
    all_modules = set()

    for state in state_dicts:
        modules = [
            key.replace(".lora_down.weight", "")
            for key in state.keys()
            if key.endswith("lora_down.weight")
        ]
        all_modules.update(modules)

    all_modules = sorted(all_modules)
    print(f"  Discovered {len(all_modules)} distinct LoRA modules")

    for module_idx, module in enumerate(all_modules):
        if module_idx % 100 == 0:
            print(f"    Processing module {module_idx}/{len(all_modules)}: {module}")

        up_chunks = []
        down_chunks = []
        total_rank = 0
        dtype = None

        for state, weight in zip(state_dicts, weights):
            down_key = f"{module}.lora_down.weight"
            up_key = f"{module}.lora_up.weight"
            alpha_key = f"{module}.alpha"

            if down_key not in state or up_key not in state:
                continue

            down = state[down_key].to(torch.float32)
            up = state[up_key].to(torch.float32)
            rank = down.shape[0]
            dtype = state[down_key].dtype

            alpha = state.get(alpha_key)
            base_alpha = alpha.item() if alpha is not None else rank
            scale = weight * (float(base_alpha) / rank)

            up_scaled = up * scale

            up_chunks.append(up_scaled)
            down_chunks.append(down)
            total_rank += rank

        if not up_chunks:
            continue

        if dtype is None:
            dtype = torch.float16

        if len(up_chunks) == 1 and module_idx % 100 == 0:
            print(f"      Only one source contributed to {module}")

        up_new = torch.cat(up_chunks, dim=1).to(dtype)
        down_new = torch.cat(down_chunks, dim=0).to(dtype)
        if module_idx % 100 == 0:
            print(
                f"      Combined rank {total_rank} for {module} "
                f"(chunks: {[chunk.shape[1] for chunk in up_chunks]})"
            )

        combined_state[f"{module}.lora_up.weight"] = up_new.to(dtype)
        combined_state[f"{module}.lora_down.weight"] = down_new.to(dtype)
        combined_state[f"{module}.alpha"] = torch.tensor(
            float(total_rank), dtype=dtype
        )

    output_file = output_dir / "pytorch_lora_weights.safetensors"
    save_file(combined_state, str(output_file))
    print(
        f"  Saved stacked LoRA with {len(combined_state)//3} modules to {output_file}"
    )
    return str(output_dir)


def main():
    # FIX FOR VERTICAL VIDEOS: Use Flash Attention instead of Video Sparse Attention
    # VIDEO_SPARSE_ATTN breaks spatial coherence for portrait orientation (tall videos)

    # Using SAGE_ATTN_THREE for ~3x speedup with acceptable quality trade-off
    # This backend is optimized for Blackwell GPUs (H200) and should work well with vertical videos
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1") 

    config = Wan2_2_T2V_A14B_Config()

    generator = VideoGenerator.from_pretrained(
        "/workspace/FastVideo/models/Wan2.2-T2V-A14B-Diffusers",
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        pipeline_config=config,
    )
    Path("/workspace/FastVideo/outputs").mkdir(parents=True, exist_ok=True)

    prompt = (
        "Instagirl, casual selfie from slightly above with the iPhone 8 front camera (7MP, f/2.2, 1080p/30), walking in Central Park at late‑afternoon golden hour; head‑and‑shoulders at arm’s length, mild wide‑angle feel, natural skin with fine pores, faint peach fuzz, tiny freckles and slight uneven tone (no beauty filter), lively eyes, subtle off‑white teeth with natural enamel translucency, a few hair flyaways, knit sweater and simple pendant; trees and a thin skyline band softly blurred behind, gentle sun flare, small hand‑held sway and a single natural blink, warm but true‑to‑life color, no oversharpening."
    )

    lora_path = "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-HIGH_converted.safetensors"
    nickname = "instagirl_high"
    lora_scale = 1.1  # defaults to 1.0
    print(f"\nApplying single LoRA {nickname} from {lora_path} with scale {lora_scale}")
    generator.set_lora_adapter(lora_nickname=nickname, lora_path=lora_path, lora_scale=lora_scale)

    negative_prompt = (
        "censored, sunburnt skin, rashy skin, red cheeks"
    )

    output_path = "/workspace/FastVideo/outputs/test_wan22_quality_fixed.mp4"
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
    )
    print(f"✓ Video generated with converted Instagirl LoRA!")


if __name__ == "__main__":
    main()
