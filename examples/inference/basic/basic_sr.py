"""
Sketch-Rendering (SR) Mode Example

SR mode uses a large model (14B) for high-noise sketching steps,
then switches to a small model (1.3B) for low-noise rendering steps.
This provides significant speedups with minimal quality loss.
"""

import os

from fastvideo import VideoGenerator


def main():
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

    generator = VideoGenerator.from_pretrained_sr(
        sketch_model_path="/workspace/models/FastWan2.1-14B-Diffusers",
        rendering_model_path="/workspace/models/FastWan2.1-1.3B-Diffusers",
        sr_diff_threshold=0.01,
        sr_min_switch_step=5,
        sr_max_switch_step=30,
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
    )

    generator.generate_video(
        prompt=
        "A serene lake at sunset with mountains in the background",
        height=720,
        width=1280,
        num_frames=81,
        num_inference_steps=50,
        save_video=True,
        output_path="sr_example_output.mp4",
    )

    print("Video generated successfully!")


if __name__ == "__main__":
    main()
