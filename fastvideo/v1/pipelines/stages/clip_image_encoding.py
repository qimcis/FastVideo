# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import torch

from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.vision_utils import load_image
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class CLIPImageEncodingStage(PipelineStage):
    """
    Stage for encoding image prompts into embeddings for diffusion models.
    
    This stage handles the encoding of image prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, image_encoder, image_processor) -> None:
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary image encoder.
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def forward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt into image encoder hidden states.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The batch with encoded prompt embeddings.
        """
        if inference_args.use_cpu_offload:
            self.image_encoder = self.image_encoder.to(batch.device)

        image = load_image(batch.image_path)

        image_inputs = self.image_processor(
            images=image, return_tensors="pt").to(batch.device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            image_embeds = self.image_encoder(**image_inputs)

        batch.image_embeds.append(image_embeds)

        if inference_args.use_cpu_offload:
            self.image_encoder.to('cpu')
            torch.cuda.empty_cache()

        return batch
