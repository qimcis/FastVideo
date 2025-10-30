import os

from fastvideo import VideoGenerator


def main():
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    generator = VideoGenerator.from_pretrained_sr(
        sketch_model_path="/workspace/models/FastWan2.1-14B-Diffusers",
        rendering_model_path="/workspace/models/FastWan2.1-1.3B-Diffusers",
        sr_diff_threshold=0.01,
        sr_min_switch_step=5,
        sr_max_switch_step=30,
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        lora_path="/workspace/FastVideo/loras",
    )
    generator.generate_video(
        prompt="early 2010s snapshot photo captured with a phone and uploaded to facebook, featuring dynamic natural lighting, and a neutral white color balance with washed out colors, casual selfie from slightly above with the iPhone 8 front camera (7MP, f/2.2, 1080p/30), walking in Central Park at late‑afternoon golden hour; head‑and‑shoulders at arm's length, mild wide‑angle feel, natural skin with fine pores, faint peach fuzz, tiny freckles and slight uneven tone (no beauty filter), lively eyes, subtle off‑white teeth with natural enamel translucency, a few hair flyaways, knit sweater and simple pendant; trees and a thin skyline band softly blurred behind, gentle sun flare, small hand‑held sway and a single natural blink, warm but true‑to‑life color, no oversharpening.",
        height=1280,
        width=720,
        num_frames=150,
        num_inference_steps=20,
        fps=30,
        guidance_scale=2.5,
        save_video=True,
        output_path="test_sr_multi_gpu.mp4",
    )

    print("Generation successful!")


if __name__ == "__main__":
    main()
