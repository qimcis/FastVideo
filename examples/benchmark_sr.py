import time

import torch

from fastvideo import VideoGenerator


def run_prompt(generator: VideoGenerator, prompt: str,
               **kwargs) -> float:
    start = time.time()
    generator.generate_video(prompt=prompt, **kwargs)
    torch.cuda.empty_cache()
    return time.time() - start


def main():
    prompts = [
        "A cat playing with a ball",
        "Sunset over mountains",
        "City traffic at night",
    ]

    baseline_generator = VideoGenerator.from_pretrained(
        "/workspace/models/FastWan2.1-14B-Diffusers",
        num_gpus=2,
    )

    baseline_times = [
        run_prompt(baseline_generator, prompt, num_inference_steps=50)
        for prompt in prompts
    ]

    sr_generator = VideoGenerator.from_pretrained_sr(
        sketch_model_path="/workspace/models/FastWan2.1-14B-Diffusers",
        rendering_model_path="/workspace/models/FastWan2.1-1.3B-Diffusers",
        num_gpus=2,
    )

    sr_times = [
        run_prompt(sr_generator, prompt, num_inference_steps=50)
        for prompt in prompts
    ]

    baseline_avg = sum(baseline_times) / len(baseline_times)
    sr_avg = sum(sr_times) / len(sr_times)
    speedup = sum(baseline_times) / sum(sr_times)

    print(f"Baseline avg: {baseline_avg:.2f}s")
    print(f"SR avg: {sr_avg:.2f}s")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
