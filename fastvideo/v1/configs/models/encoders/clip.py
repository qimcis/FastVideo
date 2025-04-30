from dataclasses import dataclass
from typing import Optional

from fastvideo.v1.configs.models.encoders.base import (ImageEncoderArchConfig,
                                                       ImageEncoderConfig,
                                                       TextEncoderArchConfig,
                                                       TextEncoderConfig)


@dataclass
class CLIPTextArchConfig(TextEncoderArchConfig):
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    text_len: int = 77


@dataclass
class CLIPVisionArchConfig(ImageEncoderArchConfig):
    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 32
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0


@dataclass
class CLIPTextConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = CLIPTextArchConfig()

    num_hidden_layers_override: Optional[int] = None
    require_post_norm: Optional[bool] = None
    prefix: str = "clip"


@dataclass
class CLIPVisionConfig(ImageEncoderConfig):
    arch_config: ImageEncoderArchConfig = CLIPVisionArchConfig()

    num_hidden_layers_override: Optional[int] = None
    require_post_norm: Optional[bool] = None
    prefix: str = "clip"
