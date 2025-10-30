# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Hashable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.layers.lora.linear import (BaseLayerWithLoRA, get_lora_layer,
                                          replace_submodule)
from fastvideo.logger import init_logger
from fastvideo.models.loader.utils import get_param_names_mapping
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.utils import maybe_download_lora

logger = init_logger(__name__)


class LoRAPipeline(ComposedPipelineBase):
    """
    Pipeline that supports injecting LoRA adapters into the diffusion transformer.
    TODO: support training.
    """
    lora_adapters: dict[str, dict[str, torch.Tensor]] = defaultdict(
        dict)  # state dicts of loaded lora adapters
    cur_adapter_name: str = ""
    cur_adapter_path: str = ""
    lora_layers: dict[str, BaseLayerWithLoRA] = {}
    lora_layers_critic: dict[str, BaseLayerWithLoRA] = {}
    fastvideo_args: FastVideoArgs | TrainingArgs
    exclude_lora_layers: list[str] = []
    device: torch.device = get_local_torch_device()
    lora_target_modules: list[str] | None = None
    lora_path: str | None = None
    lora_nickname: str = "default"
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_initialized: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = get_local_torch_device()
        self.exclude_lora_layers = self.modules[
            "transformer"].config.arch_config.exclude_lora_layers
        self.lora_target_modules = self.fastvideo_args.lora_target_modules
        self.lora_path = self.fastvideo_args.lora_path
        self.lora_nickname = self.fastvideo_args.lora_nickname
        self.training_mode = self.fastvideo_args.training_mode
        if self.training_mode and getattr(self.fastvideo_args, "lora_training",
                                          False):
            assert isinstance(self.fastvideo_args, TrainingArgs)
            if self.fastvideo_args.lora_alpha is None:
                self.fastvideo_args.lora_alpha = self.fastvideo_args.lora_rank
            self.lora_rank = self.fastvideo_args.lora_rank  # type: ignore
            self.lora_alpha = self.fastvideo_args.lora_alpha  # type: ignore
            logger.info("Using LoRA training with rank %d and alpha %d",
                        self.lora_rank, self.lora_alpha)
            if self.lora_target_modules is None:
                self.lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj", "to_q", "to_k",
                    "to_v", "to_out", "to_qkv"
                ]
            self.convert_to_lora_layers()
        # Inference
        elif not self.training_mode and self.lora_path is not None:
            self.convert_to_lora_layers()
            self.set_lora_adapter(
                self.lora_nickname,  # type: ignore
                self.lora_path)  # type: ignore

    def is_target_layer(self, module_name: str) -> bool:
        if self.lora_target_modules is None:
            return True
        return any(target_name in module_name
                   for target_name in self.lora_target_modules)

    def set_trainable(self) -> None:

        def set_lora_grads(lora_layers: dict[str, BaseLayerWithLoRA],
                           device_mesh: DeviceMesh):
            for name, layer in lora_layers.items():
                layer.lora_A.requires_grad_(True)
                layer.lora_B.requires_grad_(True)
                layer.base_layer.requires_grad_(False)
                layer.lora_A = nn.Parameter(
                    DTensor.from_local(layer.lora_A, device_mesh=device_mesh))
                layer.lora_B = nn.Parameter(
                    DTensor.from_local(layer.lora_B, device_mesh=device_mesh))

        is_lora_training = self.training_mode and getattr(
            self.fastvideo_args, "lora_training", False)
        if not is_lora_training:
            super().set_trainable()
            return

        self.modules["transformer"].requires_grad_(False)
        if "fake_score_transformer" in self.modules:
            self.modules["fake_score_transformer"].requires_grad_(False)
        device_mesh = init_device_mesh("cuda", (dist.get_world_size(), 1),
                                       mesh_dim_names=["fake", "replicate"])
        set_lora_grads(self.lora_layers, device_mesh)
        set_lora_grads(self.lora_layers_critic, device_mesh)

    def convert_to_lora_layers(self) -> None:
        """
        Unified method to convert the transformer to a LoRA transformer.
        Also converts transformer_2 if present (for MoE models like Wan2.2).
        Separate LoRAs can be applied to each transformer.
        """
        if self.lora_initialized:
            return
        self.lora_initialized = True

        # Convert transformer (high noise expert)
        converted_count = 0
        for name, layer in self.modules["transformer"].named_modules():
            if not self.is_target_layer(name):
                continue

            excluded = False
            for exclude_layer in self.exclude_lora_layers:
                if exclude_layer in name:
                    excluded = True
                    break
            if excluded:
                continue

            layer = get_lora_layer(layer,
                                   lora_rank=self.lora_rank,
                                   lora_alpha=self.lora_alpha,
                                   training_mode=self.training_mode)
            if layer is not None:
                self.lora_layers[name] = layer
                replace_submodule(self.modules["transformer"], name, layer)
                converted_count += 1
        logger.info("Converted %d layers to LoRA layers in transformer", converted_count)

        # Convert transformer_2 (low noise expert) if it exists
        if "transformer_2" in self.modules and self.modules["transformer_2"] is not None:
            converted_count_2 = 0
            for name, layer in self.modules["transformer_2"].named_modules():
                if not self.is_target_layer(name):
                    continue

                excluded = False
                for exclude_layer in self.exclude_lora_layers:
                    if exclude_layer in name:
                        excluded = True
                        break
                if excluded:
                    continue

                layer = get_lora_layer(layer,
                                       lora_rank=self.lora_rank,
                                       lora_alpha=self.lora_alpha,
                                       training_mode=self.training_mode)
                if layer is not None:
                    # Use "transformer_2" prefix to distinguish from transformer
                    self.lora_layers[f"transformer_2.{name}"] = layer
                    replace_submodule(self.modules["transformer_2"], name, layer)
                    converted_count_2 += 1
            logger.info("Converted %d layers to LoRA layers in transformer_2", converted_count_2)

        if "fake_score_transformer" in self.modules:
            for name, layer in self.modules[
                    "fake_score_transformer"].named_modules():
                if not self.is_target_layer(name):
                    continue
                layer = get_lora_layer(layer,
                                       lora_rank=self.lora_rank,
                                       lora_alpha=self.lora_alpha,
                                       training_mode=self.training_mode)
                if layer is not None:
                    self.lora_layers_critic[name] = layer
                    replace_submodule(self.modules["fake_score_transformer"],
                                      name, layer)
                    converted_count += 1
            logger.info(
                "Converted %d layers to LoRA layers in the critic model",
                converted_count)

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None,
                         lora_scale: float = 1.0):  # type: ignore
        """
        Load a LoRA adapter into the pipeline and merge it into the transformer.
        Args:
            lora_nickname: The "nick name" of the adapter when referenced in the pipeline.
            lora_path: The path to the adapter, either a local path or a Hugging Face repo id.
            lora_scale: Strength/scale of the LoRA adapter (0.0 to 2.0, default 1.0).
                       Lower values reduce LoRA influence, higher values increase it.
        """

        if lora_nickname not in self.lora_adapters and lora_path is None:
            raise ValueError(
                f"Adapter {lora_nickname} not found in the pipeline. Please provide lora_path to load it."
            )
        if not self.lora_initialized:
            rank = dist.get_rank()
            print(f"[LoRA DEBUG] rank {rank}: convert_to_lora_layers starting",
                  flush=True)
            self.convert_to_lora_layers()
            print(f"[LoRA DEBUG] rank {rank}: convert_to_lora_layers done",
                  flush=True)
        adapter_updated = False
        rank = dist.get_rank()
        print(f"[LoRA DEBUG] rank {rank}: set_lora_adapter start for "
              f"{lora_nickname} (path={lora_path})",
              flush=True)
        if lora_path is not None and lora_path != self.cur_adapter_path:
            lora_local_path = maybe_download_lora(lora_path)
            print(f"[LoRA DEBUG] rank {rank}: resolved adapter path -> "
                  f"{lora_local_path}",
                  flush=True)
            lora_state_dict = load_file(lora_local_path)
            print(f"[LoRA DEBUG] rank {rank}: loaded state dict with "
                  f"{len(lora_state_dict)} tensors",
                  flush=True)

            # Map the hf layer names to our custom layer names
            param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"].param_names_mapping)
            lora_param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"].lora_param_names_mapping)

            to_merge_params: defaultdict[Hashable,
                                         dict[Any, Any]] = defaultdict(dict)
            for idx, (name, weight) in enumerate(lora_state_dict.items()):
                name = name.replace("diffusion_model.", "")
                name = name.replace(".weight", "")
                name, _, _ = lora_param_names_mapping_fn(name)
                target_name, merge_index, num_params_to_merge = param_names_mapping_fn(
                    name)
                # for (in_dim, r) @ (r, out_dim), we only merge (r, out_dim * n) where n is the number of linear layers to fuse
                # see param mapping in HunyuanVideoArchConfig
                if merge_index is not None and "lora_B" in name:
                    to_merge_params[target_name][merge_index] = weight
                    if len(to_merge_params[target_name]) == num_params_to_merge:
                        # cat at output dim according to the merge_index order
                        sorted_tensors = [
                            to_merge_params[target_name][i]
                            for i in range(num_params_to_merge)
                        ]
                        weight = torch.cat(sorted_tensors, dim=1)
                        del to_merge_params[target_name]
                    else:
                        continue

                if target_name in self.lora_adapters[lora_nickname]:
                    raise ValueError(
                        f"Target name {target_name} already exists in lora_adapters[{lora_nickname}]"
                    )
                self.lora_adapters[lora_nickname][target_name] = weight.to(
                    self.device)
                if (idx + 1) % 500 == 0:
                    print(f"[LoRA DEBUG] rank {rank}: processed {idx+1} / "
                          f"{len(lora_state_dict)} tensors",
                          flush=True)
            adapter_updated = True
            self.cur_adapter_path = lora_path
            print(f"[LoRA DEBUG] rank {rank}: adapter tensors registered",
                  flush=True)

        if not adapter_updated and self.cur_adapter_name == lora_nickname:
            return
        self.cur_adapter_name = lora_nickname

        # Synchronize all ranks before merging to avoid FSDP deadlock
        if dist.is_initialized():
            print(f"[LoRA DEBUG] rank {rank}: waiting at barrier before merge",
                  flush=True)
            dist.barrier()
            print(f"[LoRA DEBUG] rank {rank}: passed barrier, starting merge",
                  flush=True)

        # Merge the new adapter
        adapted_count = 0
        for idx, (name, layer) in enumerate(self.lora_layers.items()):
            # For MoE models: transformer_2 layers should use the same LoRA weights
            # Strip transformer_2 prefix to find matching weights
            lookup_name = name.replace("transformer_2.", "")

            lora_A_name = lookup_name + ".lora_A"
            lora_B_name = lookup_name + ".lora_B"
            if lora_A_name in self.lora_adapters[lora_nickname]\
                and lora_B_name in self.lora_adapters[lora_nickname]:
                layer.lora_scale = lora_scale  # Set the scale before merging
                layer.set_lora_weights(
                    self.lora_adapters[lora_nickname][lora_A_name],
                    self.lora_adapters[lora_nickname][lora_B_name],
                    training_mode=self.fastvideo_args.training_mode,
                    lora_path=lora_path)
                adapted_count += 1
                if adapted_count % 200 == 0:
                    print(f"[LoRA DEBUG] rank {rank}: applied weights to "
                          f"{adapted_count}/{len(self.lora_layers)} layers",
                          flush=True)
            else:
                if rank == 0:
                    logger.warning(
                        "LoRA adapter %s does not contain the weights for layer %s. LoRA will not be applied to it.",
                        lora_path, lookup_name)
                layer.disable_lora = True

        # Synchronize all ranks after merging
        if dist.is_initialized():
            print(f"[LoRA DEBUG] rank {rank}: waiting at barrier after merge",
                  flush=True)
            dist.barrier()
            print(f"[LoRA DEBUG] rank {rank}: passed barrier after merge",
                  flush=True)

        logger.info("Rank %d: LoRA adapter %s applied to %d layers", rank,
                    lora_path, adapted_count)

    def set_dual_lora_adapters(
        self,
        lora_high_nickname: str,
        lora_high_path: str,
        lora_low_nickname: str,
        lora_low_path: str,
        lora_scale: float = 1.0
    ):
        """
        Load two separate LoRA adapters for MoE models:
        - HIGH LoRA for transformer (high noise expert, steps 0-3)
        - LOW LoRA for transformer_2 (low noise expert, steps 4-11)

        Args:
            lora_high_nickname: Nickname for HIGH LoRA
            lora_high_path: Path to HIGH LoRA safetensors file
            lora_low_nickname: Nickname for LOW LoRA
            lora_low_path: Path to LOW LoRA safetensors file
            lora_scale: Scale for both LoRAs (default 1.0)
        """
        if not self.lora_initialized:
            rank = dist.get_rank()
            print(f"[LoRA DEBUG] rank {rank}: convert_to_lora_layers starting", flush=True)
            self.convert_to_lora_layers()
            print(f"[LoRA DEBUG] rank {rank}: convert_to_lora_layers done", flush=True)

        rank = dist.get_rank()
        print(f"[LoRA DEBUG] rank {rank}: Loading dual LoRAs - HIGH and LOW", flush=True)

        # Load HIGH LoRA
        lora_high_local = maybe_download_lora(lora_high_path)
        lora_high_state = load_file(lora_high_local)
        print(f"[LoRA DEBUG] rank {rank}: Loaded HIGH LoRA with {len(lora_high_state)} tensors", flush=True)

        # Load LOW LoRA
        lora_low_local = maybe_download_lora(lora_low_path)
        lora_low_state = load_file(lora_low_local)
        print(f"[LoRA DEBUG] rank {rank}: Loaded LOW LoRA with {len(lora_low_state)} tensors", flush=True)

        # Register both LoRAs
        self._register_lora_state_dict(lora_high_nickname, lora_high_state)
        self._register_lora_state_dict(lora_low_nickname, lora_low_state)

        # Synchronize before merging
        if dist.is_initialized():
            print(f"[LoRA DEBUG] rank {rank}: waiting at barrier before dual merge", flush=True)
            dist.barrier()
            print(f"[LoRA DEBUG] rank {rank}: passed barrier, starting dual merge", flush=True)

        # Merge LoRAs to appropriate transformers
        adapted_count_high = 0
        adapted_count_low = 0

        for idx, (name, layer) in enumerate(self.lora_layers.items()):
            # Determine which LoRA to use based on transformer
            if name.startswith("transformer_2."):
                # LOW LoRA for transformer_2 (low noise expert)
                lora_nickname = lora_low_nickname
                lora_path_used = lora_low_path
                lookup_name = name.replace("transformer_2.", "")
                is_transformer_2 = True
            else:
                # HIGH LoRA for transformer (high noise expert)
                lora_nickname = lora_high_nickname
                lora_path_used = lora_high_path
                lookup_name = name
                is_transformer_2 = False

            lora_A_name = lookup_name + ".lora_A"
            lora_B_name = lookup_name + ".lora_B"

            if lora_A_name in self.lora_adapters[lora_nickname] and lora_B_name in self.lora_adapters[lora_nickname]:
                layer.lora_scale = lora_scale
                layer.set_lora_weights(
                    self.lora_adapters[lora_nickname][lora_A_name],
                    self.lora_adapters[lora_nickname][lora_B_name],
                    training_mode=self.fastvideo_args.training_mode,
                    lora_path=lora_path_used
                )
                if is_transformer_2:
                    adapted_count_low += 1
                else:
                    adapted_count_high += 1

                if (adapted_count_high + adapted_count_low) % 200 == 0:
                    print(f"[LoRA DEBUG] rank {rank}: applied weights to "
                          f"{adapted_count_high + adapted_count_low}/{len(self.lora_layers)} layers "
                          f"(HIGH: {adapted_count_high}, LOW: {adapted_count_low})", flush=True)
            else:
                if rank == 0:
                    logger.warning(
                        "LoRA adapter %s does not contain weights for layer %s. LoRA will not be applied.",
                        lora_nickname, lookup_name
                    )
                layer.disable_lora = True

        # Synchronize after merging
        if dist.is_initialized():
            print(f"[LoRA DEBUG] rank {rank}: waiting at barrier after dual merge", flush=True)
            dist.barrier()
            print(f"[LoRA DEBUG] rank {rank}: passed barrier after dual merge", flush=True)

        logger.info("Rank %d: Applied dual LoRAs - HIGH: %d layers, LOW: %d layers",
                    rank, adapted_count_high, adapted_count_low)

    def _register_lora_state_dict(self, lora_nickname: str, lora_state_dict: dict):
        """Helper method to register a LoRA state dict into the adapters."""
        rank = dist.get_rank()

        # Map the hf layer names to our custom layer names
        param_names_mapping_fn = get_param_names_mapping(
            self.modules["transformer"].param_names_mapping)
        lora_param_names_mapping_fn = get_param_names_mapping(
            self.modules["transformer"].lora_param_names_mapping)

        to_merge_params = defaultdict(dict)
        for idx, (name, weight) in enumerate(lora_state_dict.items()):
            name = name.replace("diffusion_model.", "")
            name = name.replace(".weight", "")
            name, _, _ = lora_param_names_mapping_fn(name)
            target_name, merge_index, num_params_to_merge = param_names_mapping_fn(name)

            if merge_index is not None and "lora_B" in name:
                to_merge_params[target_name][merge_index] = weight
                if len(to_merge_params[target_name]) == num_params_to_merge:
                    sorted_tensors = [
                        to_merge_params[target_name][i]
                        for i in range(num_params_to_merge)
                    ]
                    weight = torch.cat(sorted_tensors, dim=1)
                    del to_merge_params[target_name]
                else:
                    continue

            if target_name in self.lora_adapters[lora_nickname]:
                raise ValueError(
                    f"Target name {target_name} already exists in lora_adapters[{lora_nickname}]"
                )
            self.lora_adapters[lora_nickname][target_name] = weight.to(self.device)

        print(f"[LoRA DEBUG] rank {rank}: registered {len(self.lora_adapters[lora_nickname])} tensors for {lora_nickname}", flush=True)

    def merge_lora_weights(self) -> None:
        for name, layer in self.lora_layers.items():
            layer.merge_lora_weights()

    def unmerge_lora_weights(self) -> None:
        for name, layer in self.lora_layers.items():
            layer.unmerge_lora_weights()
