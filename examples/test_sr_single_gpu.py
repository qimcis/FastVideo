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
        num_gpus=1,
        dit_cpu_offload=True,
    )

    generator.generate_video(
        prompt="A cat walking in a garden",
        height=480,
        width=832,
        num_frames=49,
        num_inference_steps=30,
        save_video=True,
        output_path="test_sr_output.mp4",
    )

    print("Generation successful!")


if __name__ == "__main__":
    main()
