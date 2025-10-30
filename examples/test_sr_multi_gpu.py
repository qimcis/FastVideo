import os

from fastvideo import VideoGenerator


def main():
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    generator = VideoGenerator.from_pretrained_sr(
        sketch_model_path="/workspace/models/FastWan2.1-14B-Diffusers",
        rendering_model_path="/workspace/models/FastWan2.1-1.3B-Diffusers",
        sr_diff_threshold=0.01,
        num_gpus=2,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
    )

    generator.generate_video(
        prompt="Blonde girl walking in central park",
        height=720,
        width=1280,
        num_frames=81,
        num_inference_steps=50,
        save_video=True,
        output_path="test_sr_multi_gpu.mp4",
    )

    print("Generation successful!")


if __name__ == "__main__":
    main()
