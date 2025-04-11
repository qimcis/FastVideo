# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from fastvideo.v1.attention import DistributedAttention, LocalAttention
from fastvideo.v1.distributed.parallel_state import (
    get_sequence_model_parallel_world_size)
from fastvideo.v1.layers.layernorm import (LayerNormScaleShift, ScaleResidual,
                                           ScaleResidualLayerNormScaleShift)
from fastvideo.v1.layers.linear import ReplicatedLinear
# TODO(will-PY-refactor): RMSNorm ....
from fastvideo.v1.layers.mlp import MLP
from fastvideo.v1.layers.rotary_embedding import (_apply_rotary_emb,
                                                  get_rotary_pos_embed)
from fastvideo.v1.layers.visual_embedding import (ModulateProjection,
                                                  PatchEmbed, TimestepEmbedder,
                                                  unpatchify)
from fastvideo.v1.models.dits.base import BaseDiT


class HunyuanRMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x) -> torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal DiT block with separate modulation for text and image/video,
    using distributed attention and linear layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.deterministic = False
        self.num_attention_heads = num_attention_heads
        head_dim = hidden_size // num_attention_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Image modulation components
        self.img_mod = ModulateProjection(
            hidden_size,
            factor=6,
            act_layer="silu",
            dtype=dtype,
        )

        # Fused operations for image stream
        self.img_attn_norm = LayerNormScaleShift(hidden_size,
                                                 norm_type="layer",
                                                 elementwise_affine=False,
                                                 dtype=dtype)
        self.img_attn_residual_mlp_norm = ScaleResidualLayerNormScaleShift(
            hidden_size,
            norm_type="layer",
            elementwise_affine=False,
            dtype=dtype)
        self.img_mlp_residual = ScaleResidual()

        # Image attention components
        self.img_attn_qkv = ReplicatedLinear(hidden_size,
                                             hidden_size * 3,
                                             bias=True,
                                             params_dtype=dtype)

        self.img_attn_q_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.img_attn_k_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)

        self.img_attn_proj = ReplicatedLinear(hidden_size,
                                              hidden_size,
                                              bias=True,
                                              params_dtype=dtype)

        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype)

        # Text modulation components
        self.txt_mod = ModulateProjection(
            hidden_size,
            factor=6,
            act_layer="silu",
            dtype=dtype,
        )

        # Fused operations for text stream
        self.txt_attn_norm = LayerNormScaleShift(hidden_size,
                                                 norm_type="layer",
                                                 elementwise_affine=False,
                                                 dtype=dtype)
        self.txt_attn_residual_mlp_norm = ScaleResidualLayerNormScaleShift(
            hidden_size,
            norm_type="layer",
            elementwise_affine=False,
            dtype=dtype)
        self.txt_mlp_residual = ScaleResidual()

        # Text attention components
        self.txt_attn_qkv = ReplicatedLinear(hidden_size,
                                             hidden_size * 3,
                                             bias=True,
                                             params_dtype=dtype)

        # QK norm layers for text
        self.txt_attn_q_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.txt_attn_k_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)

        self.txt_attn_proj = ReplicatedLinear(hidden_size,
                                              hidden_size,
                                              bias=True,
                                              params_dtype=dtype)

        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype)

        # Distributed attention
        self.attn = DistributedAttention(num_heads=num_attention_heads,
                                         head_size=head_dim,
                                         dropout_rate=0.0,
                                         causal=False)

        # QK norm layers for text
        self.txt_attn_q_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.txt_attn_k_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)

        self.txt_attn_proj = ReplicatedLinear(hidden_size,
                                              hidden_size,
                                              bias=True,
                                              params_dtype=dtype)

        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype)

        # Distributed attention
        self.attn = DistributedAttention(num_heads=num_attention_heads,
                                         head_size=head_dim,
                                         dropout_rate=0.0,
                                         causal=False)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process modulation vectors
        img_mod_outputs = self.img_mod(vec)
        (
            img_attn_shift,
            img_attn_scale,
            img_attn_gate,
            img_mlp_shift,
            img_mlp_scale,
            img_mlp_gate,
        ) = torch.chunk(img_mod_outputs, 6, dim=-1)

        txt_mod_outputs = self.txt_mod(vec)
        (
            txt_attn_shift,
            txt_attn_scale,
            txt_attn_gate,
            txt_mlp_shift,
            txt_mlp_scale,
            txt_mlp_gate,
        ) = torch.chunk(txt_mod_outputs, 6, dim=-1)

        # Prepare image for attention using fused operation
        img_attn_input = self.img_attn_norm(img, img_attn_shift, img_attn_scale)
        # Get QKV for image
        img_qkv, _ = self.img_attn_qkv(img_attn_input)
        batch_size, image_seq_len = img_qkv.shape[0], img_qkv.shape[1]

        # Split QKV
        img_qkv = img_qkv.view(batch_size, image_seq_len, 3,
                               self.num_attention_heads, -1)
        img_q, img_k, img_v = img_qkv[:, :, 0], img_qkv[:, :, 1], img_qkv[:, :,
                                                                          2]

        # Apply QK-Norm if needed

        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        # Apply rotary embeddings
        cos, sin = freqs_cis
        img_q, img_k = _apply_rotary_emb(
            img_q, cos, sin,
            is_neox_style=False), _apply_rotary_emb(img_k,
                                                    cos,
                                                    sin,
                                                    is_neox_style=False)
        # Prepare text for attention using fused operation
        txt_attn_input = self.txt_attn_norm(txt, txt_attn_shift, txt_attn_scale)

        # Get QKV for text
        txt_qkv, _ = self.txt_attn_qkv(txt_attn_input)
        batch_size, text_seq_len = txt_qkv.shape[0], txt_qkv.shape[1]

        # Split QKV
        txt_qkv = txt_qkv.view(batch_size, text_seq_len, 3,
                               self.num_attention_heads, -1)
        txt_q, txt_k, txt_v = txt_qkv[:, :, 0], txt_qkv[:, :, 1], txt_qkv[:, :,
                                                                          2]

        # Apply QK-Norm if needed
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_q.dtype)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_k.dtype)

        # Run distributed attention
        img_attn, txt_attn = self.attn(img_q, img_k, img_v, txt_q, txt_k, txt_v)
        img_attn_out, _ = self.img_attn_proj(
            img_attn.view(batch_size, image_seq_len, -1))
        # Use fused operation for residual connection, normalization, and modulation
        img_mlp_input, img_residual = self.img_attn_residual_mlp_norm(
            img, img_attn_out, img_attn_gate, img_mlp_shift, img_mlp_scale)

        # Process image MLP
        img_mlp_out = self.img_mlp(img_mlp_input)
        img = self.img_mlp_residual(img_residual, img_mlp_out, img_mlp_gate)

        # Process text attention output
        txt_attn_out, _ = self.txt_attn_proj(
            txt_attn.reshape(batch_size, text_seq_len, -1))

        # Use fused operation for residual connection, normalization, and modulation
        txt_mlp_input, txt_residual = self.txt_attn_residual_mlp_norm(
            txt, txt_attn_out, txt_attn_gate, txt_mlp_shift, txt_mlp_scale)

        # Process text MLP
        txt_mlp_out = self.txt_mlp(txt_mlp_input)
        txt = self.txt_mlp_residual(txt_residual, txt_mlp_out, txt_mlp_gate)

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers using distributed attention
    and tensor parallelism.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float = 4.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        head_dim = hidden_size // num_attention_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        # Combined QKV and MLP input projection
        self.linear1 = ReplicatedLinear(hidden_size,
                                        hidden_size * 3 + mlp_hidden_dim,
                                        bias=True,
                                        params_dtype=dtype)

        # Combined projection and MLP output
        self.linear2 = ReplicatedLinear(hidden_size + mlp_hidden_dim,
                                        hidden_size,
                                        bias=True,
                                        params_dtype=dtype)

        # QK norm layers
        self.q_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.k_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)

        # Fused operations with better naming
        self.input_norm_scale_shift = LayerNormScaleShift(
            hidden_size,
            norm_type="layer",
            eps=1e-6,
            elementwise_affine=False,
            dtype=dtype)
        self.output_residual = ScaleResidual()

        # Activation function
        self.mlp_act = nn.GELU(approximate="tanh")

        # Modulation
        self.modulation = ModulateProjection(hidden_size,
                                             factor=3,
                                             act_layer="silu",
                                             dtype=dtype)

        # Distributed attention
        self.attn = DistributedAttention(num_heads=num_attention_heads,
                                         head_size=head_dim,
                                         dropout_rate=0.0,
                                         causal=False)

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Process modulation
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)

        # Apply pre-norm and modulation using fused operation
        x_mod = self.input_norm_scale_shift(x, mod_shift, mod_scale)

        # Get combined projections
        linear1_out, _ = self.linear1(x_mod)

        # Split into QKV and MLP parts
        qkv, mlp = torch.split(linear1_out,
                               [3 * self.hidden_size, self.mlp_hidden_dim],
                               dim=-1)

        # Process QKV
        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_attention_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply QK-Norm
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # Split into image and text parts
        img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
        img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]
        img_v, txt_v = v[:, :-txt_len], v[:, -txt_len:]
        # Apply rotary embeddings to image parts
        cos, sin = freqs_cis
        img_q, img_k = _apply_rotary_emb(
            img_q, cos, sin,
            is_neox_style=False), _apply_rotary_emb(img_k,
                                                    cos,
                                                    sin,
                                                    is_neox_style=False)

        # Run distributed attention
        img_attn_output, txt_attn_output = self.attn(img_q, img_k, img_v, txt_q,
                                                     txt_k, txt_v)
        attn_output = torch.cat((img_attn_output, txt_attn_output),
                                dim=1).view(batch_size, seq_len, -1)
        # Process MLP activation
        mlp_output = self.mlp_act(mlp)

        # Combine attention and MLP outputs
        combined = torch.cat((attn_output, mlp_output), dim=-1)

        # Final projection
        output, _ = self.linear2(combined)

        # Apply residual connection with gating using fused operation
        return self.output_residual(x, output, mod_gate)


class HunyuanVideoTransformer3DModel(BaseDiT):
    """
    HunyuanVideo Transformer backbone adapted for distributed training.
    
    This implementation uses distributed attention and linear layers for efficient
    parallel processing across multiple GPUs.
    
    Based on the architecture from:
    - Flux.1: https://github.com/black-forest-labs/flux
    - MMDiT: http://arxiv.org/abs/2403.03206
    """
    # PY: we make the input args the same as HF config

    # shard single stream, double stream blocks, and refiner_blocks
    _fsdp_shard_conditions = [
        lambda n, m: "double" in n and str.isdigit(n.split(".")[-1]),
        lambda n, m: "single" in n and str.isdigit(n.split(".")[-1]),
        lambda n, m: "refiner" in n and str.isdigit(n.split(".")[-1]),
    ]
    _param_names_mapping = {
        # 1. context_embedder.time_text_embed submodules (specific rules, applied first):
        r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_1\.(.*)$":
        r"txt_in.t_embedder.mlp.fc_in.\1",
        r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_2\.(.*)$":
        r"txt_in.t_embedder.mlp.fc_out.\1",
        r"^context_embedder\.proj_in\.(.*)$":
        r"txt_in.input_embedder.\1",
        r"^context_embedder\.time_text_embed\.text_embedder\.linear_1\.(.*)$":
        r"txt_in.c_embedder.fc_in.\1",
        r"^context_embedder\.time_text_embed\.text_embedder\.linear_2\.(.*)$":
        r"txt_in.c_embedder.fc_out.\1",
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm1\.(.*)$":
        r"txt_in.refiner_blocks.\1.norm1.\2",
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm2\.(.*)$":
        r"txt_in.refiner_blocks.\1.norm2.\2",
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_q\.(.*)$":
        (r"txt_in.refiner_blocks.\1.self_attn_qkv.\2", 0, 3),
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_k\.(.*)$":
        (r"txt_in.refiner_blocks.\1.self_attn_qkv.\2", 1, 3),
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_v\.(.*)$":
        (r"txt_in.refiner_blocks.\1.self_attn_qkv.\2", 2, 3),
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$":
        r"txt_in.refiner_blocks.\1.self_attn_proj.\2",
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0(?:\.proj)?\.(.*)$":
        r"txt_in.refiner_blocks.\1.mlp.fc_in.\2",
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2(?:\.proj)?\.(.*)$":
        r"txt_in.refiner_blocks.\1.mlp.fc_out.\2",
        r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm_out\.linear\.(.*)$":
        r"txt_in.refiner_blocks.\1.adaLN_modulation.linear.\2",

        # 3. x_embedder mapping:
        r"^x_embedder\.proj\.(.*)$":
        r"img_in.proj.\1",

        # 4. Top-level time_text_embed mappings:
        r"^time_text_embed\.timestep_embedder\.linear_1\.(.*)$":
        r"time_in.mlp.fc_in.\1",
        r"^time_text_embed\.timestep_embedder\.linear_2\.(.*)$":
        r"time_in.mlp.fc_out.\1",
        r"^time_text_embed\.guidance_embedder\.linear_1\.(.*)$":
        r"guidance_in.mlp.fc_in.\1",
        r"^time_text_embed\.guidance_embedder\.linear_2\.(.*)$":
        r"guidance_in.mlp.fc_out.\1",
        r"^time_text_embed\.text_embedder\.linear_1\.(.*)$":
        r"vector_in.fc_in.\1",
        r"^time_text_embed\.text_embedder\.linear_2\.(.*)$":
        r"vector_in.fc_out.\1",

        # 5. transformer_blocks mapping:
        r"^transformer_blocks\.(\d+)\.norm1\.linear\.(.*)$":
        r"double_blocks.\1.img_mod.linear.\2",
        r"^transformer_blocks\.(\d+)\.norm1_context\.linear\.(.*)$":
        r"double_blocks.\1.txt_mod.linear.\2",
        r"^transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$":
        r"double_blocks.\1.img_attn_q_norm.\2",
        r"^transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$":
        r"double_blocks.\1.img_attn_k_norm.\2",
        r"^transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$":
        (r"double_blocks.\1.img_attn_qkv.\2", 0, 3),
        r"^transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$":
        (r"double_blocks.\1.img_attn_qkv.\2", 1, 3),
        r"^transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$":
        (r"double_blocks.\1.img_attn_qkv.\2", 2, 3),
        r"^transformer_blocks\.(\d+)\.attn\.add_q_proj\.(.*)$":
        (r"double_blocks.\1.txt_attn_qkv.\2", 0, 3),
        r"^transformer_blocks\.(\d+)\.attn\.add_k_proj\.(.*)$":
        (r"double_blocks.\1.txt_attn_qkv.\2", 1, 3),
        r"^transformer_blocks\.(\d+)\.attn\.add_v_proj\.(.*)$":
        (r"double_blocks.\1.txt_attn_qkv.\2", 2, 3),
        r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$":
        r"double_blocks.\1.img_attn_proj.\2",
        # Corrected: merge attn.to_add_out into the main projection.
        r"^transformer_blocks\.(\d+)\.attn\.to_add_out\.(.*)$":
        r"double_blocks.\1.txt_attn_proj.\2",
        r"^transformer_blocks\.(\d+)\.attn\.norm_added_q\.(.*)$":
        r"double_blocks.\1.txt_attn_q_norm.\2",
        r"^transformer_blocks\.(\d+)\.attn\.norm_added_k\.(.*)$":
        r"double_blocks.\1.txt_attn_k_norm.\2",
        r"^transformer_blocks\.(\d+)\.ff\.net\.0(?:\.proj)?\.(.*)$":
        r"double_blocks.\1.img_mlp.fc_in.\2",
        r"^transformer_blocks\.(\d+)\.ff\.net\.2(?:\.proj)?\.(.*)$":
        r"double_blocks.\1.img_mlp.fc_out.\2",
        r"^transformer_blocks\.(\d+)\.ff_context\.net\.0(?:\.proj)?\.(.*)$":
        r"double_blocks.\1.txt_mlp.fc_in.\2",
        r"^transformer_blocks\.(\d+)\.ff_context\.net\.2(?:\.proj)?\.(.*)$":
        r"double_blocks.\1.txt_mlp.fc_out.\2",

        # 6. single_transformer_blocks mapping:
        r"^single_transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$":
        r"single_blocks.\1.q_norm.\2",
        r"^single_transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$":
        r"single_blocks.\1.k_norm.\2",
        r"^single_transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$":
        (r"single_blocks.\1.linear1.\2", 0, 4),
        r"^single_transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$":
        (r"single_blocks.\1.linear1.\2", 1, 4),
        r"^single_transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$":
        (r"single_blocks.\1.linear1.\2", 2, 4),
        r"^single_transformer_blocks\.(\d+)\.proj_mlp\.(.*)$":
        (r"single_blocks.\1.linear1.\2", 3, 4),
        # Corrected: map proj_out to modulation.linear rather than a separate proj_out branch.
        r"^single_transformer_blocks\.(\d+)\.proj_out\.(.*)$":
        r"single_blocks.\1.linear2.\2",
        r"^single_transformer_blocks\.(\d+)\.norm\.linear\.(.*)$":
        r"single_blocks.\1.modulation.linear.\2",

        # 7. Final layers mapping:
        r"^norm_out\.linear\.(.*)$":
        r"final_layer.adaLN_modulation.linear.\1",
        r"^proj_out\.(.*)$":
        r"final_layer.linear.\1",
    }

    def __init__(
            self,
            patch_size: int = 2,
            patch_size_t: int = 1,
            in_channels: int = 16,
            out_channels: int = 16,
            num_attention_heads: int = 24,
            attention_head_dim: int = 128,
            mlp_ratio: float = 4.0,
            num_layers: int = 20,
            num_single_layers: int = 40,
            num_refiner_layers: int = 2,
            rope_axes_dim: Tuple[int, int, int] = (16, 56, 56),
            guidance_embeds: bool = False,
            dtype: Optional[torch.dtype] = None,
            text_embed_dim: int = 4096,
            pooled_projection_dim: int = 768,
            rope_theta: int = 256,
            qk_norm: str = "rms_norm",  #TODO(PY)
    ):
        super().__init__()
        hidden_size = attention_head_dim * num_attention_heads
        self.patch_size = [patch_size_t, patch_size, patch_size]
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embeds = guidance_embeds
        self.rope_dim_list = list(rope_axes_dim)
        self.rope_theta = rope_theta
        self.text_states_dim = text_embed_dim
        self.text_states_dim_2 = pooled_projection_dim
        # TODO(will): hack?
        self.dtype = dtype

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}"
            )

        pe_dim = hidden_size // num_attention_heads
        if sum(rope_axes_dim) != pe_dim:
            raise ValueError(
                f"Got {rope_axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        # Image projection
        self.img_in = PatchEmbed(self.patch_size,
                                 self.in_channels,
                                 self.hidden_size,
                                 dtype=dtype)

        self.txt_in = SingleTokenRefiner(self.text_states_dim,
                                         hidden_size,
                                         num_attention_heads,
                                         depth=num_refiner_layers,
                                         dtype=dtype)

        # Time modulation
        self.time_in = TimestepEmbedder(self.hidden_size,
                                        act_layer="silu",
                                        dtype=dtype)

        # Text modulation
        self.vector_in = MLP(self.text_states_dim_2,
                             self.hidden_size,
                             self.hidden_size,
                             act_type="silu",
                             dtype=dtype)

        # Guidance modulation
        self.guidance_in = (TimestepEmbedder(
            self.hidden_size, act_layer="silu", dtype=dtype)
                            if self.guidance_embeds else None)

        # Double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                hidden_size,
                num_attention_heads,
                mlp_ratio=mlp_ratio,
                dtype=dtype,
            ) for _ in range(num_layers)
        ])

        # Single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                hidden_size,
                num_attention_heads,
                mlp_ratio=mlp_ratio,
                dtype=dtype,
            ) for _ in range(num_single_layers)
        ])

        self.final_layer = FinalLayer(hidden_size,
                                      self.patch_size,
                                      self.out_channels,
                                      dtype=dtype)

        self.__post_init__()

    # TODO: change the input the FORWAD_BACTCH Dict
    # TODO: change output to a dict
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        timestep: torch.LongTensor,
        guidance=None,
    ):
        """
        Forward pass of the HunyuanDiT model.
        
        Args:
            hidden_states: Input image/video latents [B, C, T, H, W]
            encoder_hidden_states: Text embeddings [B, L, D]
            timestep: Diffusion timestep
            guidance: Guidance scale for CFG
            
        Returns:
            Tuple of (output)
        """
        if guidance is None:
            guidance = torch.tensor([6016.0],
                                    device=hidden_states.device,
                                    dtype=hidden_states.dtype)

        img = x = hidden_states
        t = timestep

        # Split text embeddings - first token is global, rest are per-token
        if isinstance(encoder_hidden_states, torch.Tensor):
            txt = encoder_hidden_states[:, 1:]
            text_states_2 = encoder_hidden_states[:, 0, :self.text_states_dim_2]
        else:
            txt = encoder_hidden_states[0]
            text_states_2 = encoder_hidden_states[1]

        # Get spatial dimensions
        _, _, ot, oh, ow = x.shape  # codespell:ignore
        tt, th, tw = (
            ot // self.patch_size[0],  # codespell:ignore
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Get rotary embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (tt * get_sequence_model_parallel_world_size(), th, tw),
            self.hidden_size, self.num_attention_heads, self.rope_dim_list,
            self.rope_theta)
        freqs_cos = freqs_cos.to(x.device)
        freqs_sin = freqs_sin.to(x.device)
        # Prepare modulation vectors
        vec = self.time_in(t)

        # Add text modulation
        vec = vec + self.vector_in(text_states_2)

        # Add guidance modulation if needed
        if self.guidance_in and guidance is not None:
            vec = vec + self.guidance_in(guidance)
        # Embed image and text
        img = self.img_in(img)
        txt = self.txt_in(txt, t)
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # Process through double stream blocks
        for index, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, freqs_cis]
            img, txt = block(*double_block_args)
        # Merge txt and img to pass through single stream blocks
        x = torch.cat((img, txt), 1)

        # Process through single stream blocks
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    freqs_cis,
                ]
                x = block(*single_block_args)

        # Extract image features
        img = x[:, :img_seq_len, ...]
        # Final layer processing
        img = self.final_layer(img, vec)
        # Unpatchify to get original shape
        img = unpatchify(img, tt, th, tw, self.patch_size, self.out_channels)

        return img


class SingleTokenRefiner(nn.Module):
    """
    A token refiner that processes text embeddings with attention to improve
    their representation for cross-attention with image features.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        num_attention_heads,
        depth=2,
        qkv_bias=True,
        dtype=None,
    ) -> None:
        super().__init__()

        # Input projection
        self.input_embedder = ReplicatedLinear(in_channels,
                                               hidden_size,
                                               bias=True,
                                               params_dtype=dtype)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size,
                                           act_layer="silu",
                                           dtype=dtype)

        # Context embedding
        self.c_embedder = MLP(in_channels,
                              hidden_size,
                              hidden_size,
                              act_type="silu",
                              dtype=dtype)

        # Refiner blocks
        self.refiner_blocks = nn.ModuleList([
            IndividualTokenRefinerBlock(
                hidden_size,
                num_attention_heads,
                qkv_bias=qkv_bias,
                dtype=dtype,
            ) for _ in range(depth)
        ])

    def forward(self, x, t):
        # Get timestep embeddings
        timestep_aware_representations = self.t_embedder(t)

        # Get context-aware representations

        context_aware_representations = torch.mean(x, dim=1)

        context_aware_representations = self.c_embedder(
            context_aware_representations)
        c = timestep_aware_representations + context_aware_representations
        # Project input
        x, _ = self.input_embedder(x)
        # Process through refiner blocks
        for block in self.refiner_blocks:
            x = block(x, c)
        return x


class IndividualTokenRefinerBlock(nn.Module):
    """
    A transformer block for refining individual tokens with self-attention.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        dtype=None,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Normalization and attention
        self.norm1 = nn.LayerNorm(hidden_size,
                                  eps=1e-6,
                                  elementwise_affine=True,
                                  dtype=dtype)

        self.self_attn_qkv = ReplicatedLinear(hidden_size,
                                              hidden_size * 3,
                                              bias=qkv_bias,
                                              params_dtype=dtype)

        self.self_attn_proj = ReplicatedLinear(hidden_size,
                                               hidden_size,
                                               bias=qkv_bias,
                                               params_dtype=dtype)

        # MLP
        self.norm2 = nn.LayerNorm(hidden_size,
                                  eps=1e-6,
                                  elementwise_affine=True,
                                  dtype=dtype)
        self.mlp = MLP(hidden_size,
                       mlp_hidden_dim,
                       bias=True,
                       act_type="silu",
                       dtype=dtype)

        # Modulation
        self.adaLN_modulation = ModulateProjection(hidden_size,
                                                   factor=2,
                                                   act_layer="silu",
                                                   dtype=dtype)

        # Scaled dot product attention
        self.attn = LocalAttention(
            num_heads=num_attention_heads,
            head_size=hidden_size // num_attention_heads,
        )

    def forward(self, x, c):
        # Get modulation parameters
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=-1)
        # Self-attention
        norm_x = self.norm1(x)
        qkv, _ = self.self_attn_qkv(norm_x)

        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_attention_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Run scaled dot product attention
        attn_output = self.attn(q, k, v)  # [B, L, H, D]
        attn_output = attn_output.reshape(batch_size, seq_len,
                                          -1)  # [B, L, H*D]

        # Project and apply residual connection with gating
        attn_out, _ = self.self_attn_proj(attn_output)
        x = x + attn_out * gate_msa.unsqueeze(1)

        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out * gate_mlp.unsqueeze(1)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT that projects features to pixel space.
    """

    def __init__(self,
                 hidden_size,
                 patch_size,
                 out_channels,
                 dtype=None) -> None:
        super().__init__()

        # Normalization
        self.norm_final = nn.LayerNorm(hidden_size,
                                       eps=1e-6,
                                       elementwise_affine=False,
                                       dtype=dtype)

        output_dim = patch_size[0] * patch_size[1] * patch_size[2] * out_channels

        self.linear = ReplicatedLinear(hidden_size,
                                       output_dim,
                                       bias=True,
                                       params_dtype=dtype)

        # Modulation
        self.adaLN_modulation = ModulateProjection(hidden_size,
                                                   factor=2,
                                                   act_layer="silu",
                                                   dtype=dtype)

    def forward(self, x, c):
        # What the heck HF? Why you change the scale and shift order here???
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x, _ = self.linear(x)
        return x
