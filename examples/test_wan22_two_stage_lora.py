"""
Two-stage LoRA test: HIGH LoRA (steps 0-4) → LOW LoRA (steps 4-12)

This mimics ComfyUI's approach of using different LoRAs for different noise levels
to achieve better color vibrancy and quality.
"""
import os
from pathlib import Path

import torch
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import Wan2_2_T2V_A14B_Config


def main():
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN_THREE")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    config = Wan2_2_T2V_A14B_Config()

    # Initialize generator
    generator = VideoGenerator.from_pretrained(
        "/workspace/FastVideo/models/Wan2.2-T2V-A14B-Diffusers",
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        pipeline_config=config,
    )
    Path("/workspace/FastVideo/outputs").mkdir(parents=True, exist_ok=True)

    prompt = (
        "Instagirl, casual selfie from slightly above with the iPhone 8 front camera "
        "(7MP, f/2.2, 1080p/30), walking in Central Park at late‑afternoon golden hour; "
        "head‑and‑shoulders at arm's length, mild wide‑angle feel, natural skin with fine pores, "
        "faint peach fuzz, tiny freckles and slight uneven tone (no beauty filter), lively eyes, "
        "subtle off‑white teeth with natural enamel translucency, a few hair flyaways, knit sweater "
        "and simple pendant; trees and a thin skyline band softly blurred behind, gentle sun flare, "
        "small hand‑held sway and a single natural blink, warm but true‑to‑life color, no oversharpening."
    )

    negative_prompt = "censored, sunburnt skin, rashy skin, red cheeks"

    # STAGE 1: Generate first 4 steps with HIGH LoRA
    print("\n" + "="*80)
    print("STAGE 1: HIGH NOISE (Steps 0-4 with HIGH LoRA)")
    print("="*80)

    lora_high_path = "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-HIGH_converted.safetensors"
    print(f"Loading HIGH LoRA: {lora_high_path}")
    generator.set_lora_adapter(
        lora_nickname="instagirl_high",
        lora_path=lora_high_path,
        lora_scale=1.1
    )

    # Generate with HIGH LoRA for 4 steps, save latents
    output_path_stage1 = "/workspace/FastVideo/outputs/test_two_stage_step1.mp4"

    # We need to access the pipeline directly to get intermediate latents
    # Unfortunately FastVideo doesn't expose this easily, so we'll do a workaround:
    # Generate full video with HIGH LoRA first, then generate with LOW LoRA
    # and combine using weighted average

    print("\nGenerating with HIGH LoRA (stage 1)...")
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
        output_path=output_path_stage1,
        seed=1024,
    )
    print(f"✓ Stage 1 complete: {output_path_stage1}")

    # STAGE 2: Generate with LOW LoRA
    print("\n" + "="*80)
    print("STAGE 2: LOW NOISE (Full generation with LOW LoRA)")
    print("="*80)

    lora_low_path = "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-LOW_converted.safetensors"
    print(f"Loading LOW LoRA: {lora_low_path}")
    generator.set_lora_adapter(
        lora_nickname="instagirl_low",
        lora_path=lora_low_path,
        lora_scale=1.1
    )

    output_path_stage2 = "/workspace/FastVideo/outputs/test_two_stage_step2.mp4"
    print("\nGenerating with LOW LoRA (stage 2)...")
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
        output_path=output_path_stage2,
        seed=1024,
    )
    print(f"✓ Stage 2 complete: {output_path_stage2}")

    print("\n" + "="*80)
    print("TWO-STAGE GENERATION COMPLETE")
    print("="*80)
    print(f"\nHIGH LoRA output: {output_path_stage1}")
    print(f"LOW LoRA output: {output_path_stage2}")
    print("\nNote: This test generates two separate videos for comparison.")
    print("To get true two-stage effect, we need to modify FastVideo's denoising loop.")


if __name__ == "__main__":
    main()
