# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from fastvideo.v1.configs.models import DiTConfig
from fastvideo.v1.platforms import _Backend


# TODO
class BaseDiT(nn.Module, ABC):
    _fsdp_shard_conditions: list = []
    _param_names_mapping: dict
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int
    # always supports torch_sdpa
    _supported_attention_backends: Tuple[
        _Backend, ...] = DiTConfig()._supported_attention_backends

    def __init_subclass__(cls) -> None:
        required_class_attrs = [
            "_fsdp_shard_conditions", "_param_names_mapping"
        ]
        super().__init_subclass__()
        for attr in required_class_attrs:
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' class variable"
                )

    def __init__(self, config: DiTConfig, **kwargs) -> None:
        super().__init__()
        self.config = config
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: Optional[Union[
                    torch.Tensor, List[torch.Tensor]]] = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        pass

    def __post_init__(self) -> None:
        required_attrs = [
            "hidden_size", "num_attention_heads", "num_channels_latents"
        ]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' instance variable"
                )

    @property
    def supported_attention_backends(self) -> Tuple[_Backend, ...]:
        return self._supported_attention_backends


class CachableDiT(BaseDiT):
    """
    An intermediate base class that adds TeaCache optimization functionality to DiT models.
    TeaCache accelerates inference by selectively skipping redundant computation when consecutive
    diffusion steps are similar enough.
    """
    # These are required class attributes that should be overridden by concrete implementations
    _fsdp_shard_conditions = []
    _param_names_mapping = {}
    # Ensure these instance attributes are properly defined in subclasses
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int
    # always supports torch_sdpa
    _supported_attention_backends: Tuple[
        _Backend, ...] = DiTConfig()._supported_attention_backends

    def __init__(self, config: DiTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.cnt = 0
        self.teacache_thresh = 0
        self.coefficients: list[float] = []

        # NOTE(will): Only wan2.1 needs these, so we are hardcoding it here
        if self.config.prefix == "wan":
            self.use_ret_steps = self.config.cache_config.use_ret_steps
            self.is_even = False
            self.previous_e0_even: torch.Tensor | None = None
            self.previous_e0_odd: torch.Tensor | None = None
            self.previous_residual_even: torch.Tensor | None = None
            self.previous_residual_odd: torch.Tensor | None = None
            self.accumulated_rel_l1_distance_even = 0
            self.accumulated_rel_l1_distance_odd = 0
            self.should_calc_even = True
            self.should_calc_odd = True
        else:
            self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = None
            self.previous_residual = None

    def maybe_cache_states(self, hidden_states: torch.Tensor,
                           original_hidden_states: torch.Tensor) -> None:
        pass

    def should_skip_forward_for_cached_states(self,
                                              **kwargs: dict[str, Any]) -> bool:
        return False

    def retrieve_cached_states(self,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "maybe_retrieve_cached_states is not implemented")
