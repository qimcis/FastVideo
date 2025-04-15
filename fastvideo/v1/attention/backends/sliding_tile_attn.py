import json
from dataclasses import dataclass
from typing import List, Optional, Type

import torch
from einops import rearrange
from st_attn import sliding_tile_attention

import fastvideo.v1.envs as envs
from fastvideo.v1.attention.backends.abstract import (AttentionBackend,
                                                      AttentionImpl,
                                                      AttentionMetadata,
                                                      AttentionMetadataBuilder)
from fastvideo.v1.distributed import get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


# TODO(will-refactor): move this to a utils file
def dict_to_3d_list(mask_strategy,
                    t_max=50,
                    l_max=60,
                    h_max=24) -> List[List[List[Optional[torch.Tensor]]]]:
    result = [[[None for _ in range(h_max)] for _ in range(l_max)]
              for _ in range(t_max)]
    if mask_strategy is None:
        return result
    for key, value in mask_strategy.items():
        t, layer, h = map(int, key.split('_'))
        result[t][layer][h] = value
    return result


class SlidingTileAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        # TODO(will-refactor): check this
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SLIDING_TILE_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["SlidingTileAttentionImpl"]:
        return SlidingTileAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["SlidingTileAttentionMetadata"]:
        return SlidingTileAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["SlidingTileAttentionMetadataBuilder"]:
        return SlidingTileAttentionMetadataBuilder


@dataclass
class SlidingTileAttentionMetadata(AttentionMetadata):
    current_timestep: int


class SlidingTileAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep: int,
        forward_batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> SlidingTileAttentionMetadata:

        return SlidingTileAttentionMetadata(current_timestep=current_timestep, )


class SlidingTileAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: Optional[int] = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        # TODO(will-refactor): for now this is the mask strategy, but maybe we should
        # have a more general config for STA?
        config_file = envs.FASTVIDEO_ATTENTION_CONFIG
        if config_file is None:
            raise ValueError("FASTVIDEO_ATTENTION_CONFIG is not set")

        with open(config_file) as f:
            mask_strategy = json.load(f)

        mask_strategy = dict_to_3d_list(mask_strategy)
        self.prefix = prefix
        self.mask_strategy = mask_strategy
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size

    def tile(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x,
                      "b (sp t h w) head d -> b (t sp h w) head d",
                      sp=self.sp_size,
                      t=30 // self.sp_size,
                      h=48,
                      w=80)
        return rearrange(
            x,
            "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
            n_t=5,
            n_h=6,
            n_w=10,
            ts_t=6,
            ts_h=8,
            ts_w=8)

    def untile(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
            n_t=5,
            n_h=6,
            n_w=10,
            ts_t=6,
            ts_h=8,
            ts_w=8)
        return rearrange(x,
                         "b (t sp h w) head d -> b (sp t h w) head d",
                         sp=self.sp_size,
                         t=30 // self.sp_size,
                         h=48,
                         w=80)

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self.tile(qkv)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: SlidingTileAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: SlidingTileAttentionMetadata,
    ) -> torch.Tensor:

        assert self.mask_strategy is not None, "mask_strategy cannot be None for SlidingTileAttention"
        assert self.mask_strategy[
            0] is not None, "mask_strategy[0] cannot be None for SlidingTileAttention"

        timestep = attn_metadata.current_timestep
        # pattern:'.double_blocks.0.attn.impl' or '.single_blocks.0.attn.impl'
        layer_idx = int(self.prefix.split('.')[-3])
        # TODO: remove hardcode
        text_length = q.shape[1] - (30 * 48 * 80)
        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        head_num = query.size(1)
        sp_group = get_sp_group()
        current_rank = sp_group.rank_in_group
        start_head = current_rank * head_num
        windows = [
            self.mask_strategy[timestep][layer_idx][head_idx + start_head]
            for head_idx in range(head_num)
        ]
        hidden_states = sliding_tile_attention(query, key, value, windows,
                                               text_length).transpose(1, 2)

        return hidden_states
