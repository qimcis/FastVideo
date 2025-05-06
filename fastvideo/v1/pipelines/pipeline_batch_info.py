# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py
"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from fastvideo.v1.configs.sample.base import TeaCacheParams
from fastvideo.v1.configs.sample.wan import WanTeaCacheParams


@dataclass
class ForwardBatch:
    """
    Complete state passed through the pipeline execution.
    
    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """
    # TODO(will): double check that args are separate from fastvideo_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    data_type: str

    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None

    # Image inputs
    image_path: Optional[str] = None
    image_embeds: List[torch.Tensor] = field(default_factory=list)

    # Text inputs
    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    prompt_path: Optional[str] = None
    output_path: str = "outputs/"

    # Primary encoder embeddings
    prompt_embeds: List[torch.Tensor] = field(default_factory=list)
    negative_prompt_embeds: Optional[List[torch.Tensor]] = None

    # Additional text-related parameters
    max_sequence_length: Optional[int] = None
    prompt_template: Optional[Dict[str, Any]] = None
    do_classifier_free_guidance: bool = False

    # Batch info
    batch_size: Optional[int] = None
    num_videos_per_prompt: int = 1
    seed: Optional[int] = None
    seeds: Optional[List[int]] = None

    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False

    # Latent tensors
    latents: Optional[torch.Tensor] = None
    noise_pred: Optional[torch.Tensor] = None
    image_latent: Optional[torch.Tensor] = None

    # Latent dimensions
    height_latents: Optional[int] = None
    width_latents: Optional[int] = None
    num_frames: int = 1  # Default for image models

    # Original dimensions (before VAE scaling)
    height: Optional[int] = None
    width: Optional[int] = None
    fps: Optional[int] = None

    # Timesteps
    timesteps: Optional[torch.Tensor] = None
    timestep: Optional[Union[torch.Tensor, float, int]] = None
    step_index: Optional[int] = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: Optional[List[float]] = None

    n_tokens: Optional[int] = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: Dict[str, Any] = field(default_factory=dict)

    # Final output (after pipeline completion)
    output: Any = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra: Dict[str, Any] = field(default_factory=dict)

    # Misc
    save_video: bool = True
    return_frames: bool = False

    # TeaCache parameters
    enable_teacache: bool = False
    teacache_params: Optional[TeaCacheParams | WanTeaCacheParams] = None

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""

        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.guidance_scale > 1.0:
            self.do_classifier_free_guidance = True
            self.negative_prompt_embeds = []
