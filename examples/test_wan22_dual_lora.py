import os
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import WanT2V720PConfig

def main():
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    print("Loading Wan2.2-T2V-A14B with stacked LoRAs (instagirl + lenovo-ultrareal)...")

    # Create config first to override auto-detection
    config = WanT2V720PConfig()

    generator = VideoGenerator.from_pretrained(
        "/workspace/FastVideo/models/Wan2.2-T2V-A14B-Diffusers",
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        lora_path="/workspace/FastVideo/loras",
        pipeline_config=config,
    )

    print("\nGenerating 5 second video with stacked LoRAs...")
    video = generator.generate_video(
        prompt="Instagirl, l3n0v0, casual selfie from slightly above with the iPhone 8 front camera (7MP, f/2.2, 1080p/30), walking in Central Park at late‑afternoon golden hour; head‑and‑shoulders at arm's length, mild wide‑angle feel, natural skin with fine pores, faint peach fuzz, tiny freckles and slight uneven tone (no beauty filter), lively eyes, subtle off‑white teeth with natural enamel translucency, a few hair flyaways, knit sweater and simple pendant; trees and a thin skyline band softly blurred behind, gentle sun flare, small hand‑held sway and a single natural blink, warm but true‑to‑life color, no oversharpening.",
        height=720,
        width=1280,
        num_frames=150,
        num_inference_steps=8,
        fps=30,
        guidance_scale=2.5,
        save_video=True,
        output_path="/workspace/FastVideo/outputs/test_stacked_loras.mp4",
    )
    print("✓ Video generated successfully with Wan2.2-14B and both LoRAs stacked!")

if __name__ == "__main__":
    main()
