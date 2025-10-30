"""
Convert single-file high/low noise Wan2.2 models to FastVideo diffusers format.

This script:
1. Creates two model directories (high noise and low noise)
2. Symlinks shared components (VAE, text encoder, tokenizer, scheduler)
3. Converts transformer weights from safetensors to sharded diffusers format
"""
import os
import shutil
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel
from safetensors.torch import load_file


def convert_single_model(
    source_safetensors: str,
    output_dir: str,
    base_diffusers_model: str,
    model_type: str  # "high_noise" or "low_noise"
):
    """Convert a single-file safetensors model to diffusers format."""

    source_path = Path(source_safetensors)
    output_path = Path(output_dir)
    base_path = Path(base_diffusers_model)

    print(f"\n{'='*80}")
    print(f"Converting {model_type.upper()} model")
    print(f"{'='*80}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Base: {base_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy/symlink shared components (don't duplicate 50GB of shared data)
    shared_components = ['vae', 'text_encoder', 'tokenizer', 'scheduler']

    for component in shared_components:
        src = base_path / component
        dst = output_path / component

        if dst.exists():
            print(f"  ✓ {component} already exists")
            continue

        # Create symlink to save space
        try:
            os.symlink(src.absolute(), dst.absolute())
            print(f"  ✓ Symlinked {component}")
        except Exception as e:
            # If symlink fails, copy instead
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"  ✓ Copied {component}")

    # Copy model_index.json and modify it
    model_index_src = base_path / "model_index.json"
    model_index_dst = output_path / "model_index.json"
    if model_index_src.exists():
        shutil.copy2(model_index_src, model_index_dst)
        print(f"  ✓ Copied model_index.json")

    # Load the single-file transformer weights
    print(f"\n  Loading transformer weights from {source_path.name}...")
    print(f"    Size: {source_path.stat().st_size / 1024**3:.1f} GB")

    state_dict = load_file(str(source_path))
    print(f"    Loaded {len(state_dict)} keys")

    # Load transformer config from base model
    transformer_config_path = base_path / "transformer" / "config.json"
    print(f"\n  Loading transformer config from base model...")

    # Create transformer directory
    transformer_output = output_path / "transformer"
    transformer_output.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy2(transformer_config_path, transformer_output / "config.json")
    print(f"    ✓ Copied config.json")

    # Load transformer model and replace weights
    print(f"\n  Initializing transformer model...")
    transformer = WanTransformer3DModel.from_pretrained(
        base_path,
        subfolder="transformer",
        torch_dtype=torch.float16
    )

    # Load weights into transformer
    print(f"  Loading weights into transformer...")
    missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"    ⚠ Missing keys: {len(missing_keys)}")
        if len(missing_keys) < 10:
            for key in missing_keys[:10]:
                print(f"      - {key}")

    if unexpected_keys:
        print(f"    ⚠ Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) < 10:
            for key in unexpected_keys[:10]:
                print(f"      - {key}")

    # Save transformer in sharded format
    print(f"\n  Saving transformer to {transformer_output}...")
    transformer.save_pretrained(
        transformer_output,
        max_shard_size="5GB",
        safe_serialization=True
    )
    print(f"    ✓ Saved sharded transformer")

    # For MoE models, we need transformer_2 (copy from base for now)
    transformer_2_base = base_path / "transformer_2"
    if transformer_2_base.exists():
        transformer_2_output = output_path / "transformer_2"
        if not transformer_2_output.exists():
            print(f"\n  Symlinking transformer_2 from base model...")
            try:
                os.symlink(transformer_2_base.absolute(), transformer_2_output.absolute())
                print(f"    ✓ Symlinked transformer_2")
            except Exception:
                shutil.copytree(transformer_2_base, transformer_2_output)
                print(f"    ✓ Copied transformer_2")

    print(f"\n{'='*80}")
    print(f"✓ {model_type.upper()} model conversion complete!")
    print(f"{'='*80}\n")


def main():
    # Paths
    base_diffusers_model = "/workspace/FastVideo/models/Wan2.2-T2V-A14B-Diffusers"

    high_noise_safetensors = "/workspace/FastVideo/models/Wan2.2-T2V-HighLow-Split/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
    high_noise_output = "/workspace/FastVideo/models/Wan2.2-T2V-A14B-HighNoise-Diffusers"

    low_noise_safetensors = "/workspace/FastVideo/models/Wan2.2-T2V-HighLow-Split/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
    low_noise_output = "/workspace/FastVideo/models/Wan2.2-T2V-A14B-LowNoise-Diffusers"

    print("\n" + "="*80)
    print("WAN 2.2 HIGH/LOW NOISE MODEL CONVERTER")
    print("="*80)
    print("\nThis will convert single-file safetensors models to FastVideo diffusers format.")
    print("Shared components (VAE, text encoder) will be symlinked to save space.")
    print("\nExpected space usage:")
    print("  - High noise: ~28.6 GB (transformer only)")
    print("  - Low noise: ~28.6 GB (transformer only)")
    print("  - Shared components: symlinked (0 GB)")
    print("  - Total: ~57 GB\n")

    # Convert high noise model
    convert_single_model(
        source_safetensors=high_noise_safetensors,
        output_dir=high_noise_output,
        base_diffusers_model=base_diffusers_model,
        model_type="high_noise"
    )

    # Convert low noise model
    convert_single_model(
        source_safetensors=low_noise_safetensors,
        output_dir=low_noise_output,
        base_diffusers_model=base_diffusers_model,
        model_type="low_noise"
    )

    print("\n" + "="*80)
    print("✓ ALL CONVERSIONS COMPLETE")
    print("="*80)
    print(f"\nHigh noise model: {high_noise_output}")
    print(f"Low noise model: {low_noise_output}")
    print("\nYou can now use these models for two-stage generation!")


if __name__ == "__main__":
    main()
