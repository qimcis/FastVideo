# SPDX-License-Identifier: Apache-2.0
"""
Conditioning stage for diffusion pipelines.
"""

import torch

from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class ConditioningStage(PipelineStage):
    """
    Stage for applying conditioning to the diffusion process.
    
    This stage handles the application of conditioning, such as classifier-free guidance,
    to the diffusion process.
    """

    def forward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Apply conditioning to the diffusion process.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The batch with applied conditioning.
        """
        if not batch.do_classifier_free_guidance:
            return batch
        else:
            return batch

        logger.info("batch.negative_prompt_embeds: %s",
                    batch.negative_prompt_embeds)
        logger.info("do_classifier_free_guidance: %s",
                    batch.do_classifier_free_guidance)
        logger.info("cfg_scale: %s", batch.guidance_scale)

        # Ensure negative prompt embeddings are available
        assert batch.negative_prompt_embeds is not None, (
            "Negative prompt embeddings are required for classifier-free guidance"
        )

        # Concatenate primary embeddings and masks
        batch.prompt_embeds = torch.cat(
            [batch.negative_prompt_embeds, batch.prompt_embeds])
        if batch.attention_mask is not None:
            batch.attention_mask = torch.cat(
                [batch.negative_attention_mask, batch.attention_mask])

        # Concatenate secondary embeddings and masks if present
        if batch.prompt_embeds_2 is not None:
            batch.prompt_embeds_2 = torch.cat(
                [batch.negative_prompt_embeds_2, batch.prompt_embeds_2])
        if batch.attention_mask_2 is not None:
            batch.attention_mask_2 = torch.cat(
                [batch.negative_attention_mask_2, batch.attention_mask_2])

        return batch
