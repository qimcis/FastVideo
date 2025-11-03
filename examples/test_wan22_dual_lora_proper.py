"""
Proper dual LoRA test for Wan2.2 MoE model with adjusted boundary ratio.

HIGH LoRA → transformer (high noise expert, steps 1-4, 33%)
LOW LoRA → transformer_2 (low noise expert, steps 5-12, 67%)

boundary_ratio = 0.965 matches the combined LoRA weighting (33% HIGH + 67% LOW)
for maximum color vibrancy.
"""
import os
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import Wan2_2_T2V_A14B_Config


def main():
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
        "Instagirl, casual selfie from slightly above with the iPhone 8 front camera "
        "(7MP, f/2.2, 1080p/30), walking in Central Park at late‑afternoon golden hour; "
        "head‑and‑shoulders at arm's length, mild wide‑angle feel, natural skin with fine pores, "
        "faint peach fuzz, tiny freckles and slight uneven tone (no beauty filter), lively eyes, "
        "subtle off‑white teeth with natural enamel translucency, a few hair flyaways, knit sweater "
        "and simple pendant; trees and a thin skyline band softly blurred behind, gentle sun flare, "
        "small hand‑held sway and a single natural blink, warm but true‑to‑life color, no oversharpening."
    )

    negative_prompt = "censored, sunburnt skin, rashy skin, red cheeks"

    # Apply dual LoRAs (HIGH + LOW)
    lora_high_path = "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-HIGH_converted.safetensors"
    lora_low_path = "/workspace/FastVideo/loras/Instagirlv2.5/Instagirlv2.5-LOW_converted.safetensors"

    print("\n" + "="*80)
    print("APPLYING DUAL LORAS (HIGH + LOW)")
    print("="*80)
    print(f"HIGH LoRA (transformer): {lora_high_path}")
    print(f"LOW LoRA (transformer_2): {lora_low_path}")
    print("="*80 + "\n")

    generator.set_dual_lora_adapters(
        lora_high_nickname="instagirl_high",
        lora_high_path=lora_high_path,
        lora_low_nickname="instagirl_low",
        lora_low_path=lora_low_path,
        lora_scale=1.1
    )

    output_path = "/workspace/FastVideo/outputs/test_dual_lora_proper.mp4"
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
        boundary_ratio=0.965,  # 4 HIGH steps + 8 LOW steps (33%/67% split)
    )

    print("\n" + "="*80)
    print("✓ VIDEO GENERATED WITH DUAL LORAS (33% HIGH + 67% LOW)")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"boundary_ratio: 0.965 (4 HIGH steps + 8 LOW steps)")
    print("\nThis should have vibrant colors matching the combined LoRA!")
    print("The 33%/67% split matches the combined LoRA weighting.")


if __name__ == "__main__":
    main()
