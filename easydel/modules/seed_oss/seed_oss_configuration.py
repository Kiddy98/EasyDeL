# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the Seed-Oss family of models."""

import typing as tp

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


def _extract_rope_scaling(rope_parameters: dict[str, tp.Any] | None) -> dict[str, tp.Any] | None:
    """Split RoPE scaling fields from the parameters payload."""
    if rope_parameters is None:
        return None

    scaling = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
    return scaling or None


@register_config("seed_oss")
class SeedOssConfig(EasyDeLBaseConfig):
    """
    EasyDeL configuration for the Seed-Oss decoder-only models.

    Args:
        vocab_size: Size of the token embedding vocabulary.
        hidden_size: Dimensionality of hidden representations.
        intermediate_size: Dimensionality of the MLP intermediate layer.
        num_hidden_layers: Number of transformer decoder blocks.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key/value heads (for GQA/MQA support).
        hidden_act: Activation used inside the MLP (defaults to SiLU).
        max_position_embeddings: Maximum supported positional embedding index.
        initializer_range: Standard deviation for parameter initialisation.
        rms_norm_eps: Epsilon for RMSNorm layers.
        use_cache: Whether KV cache is enabled by default.
        pad_token_id: Padding token identifier.
        bos_token_id: Beginning-of-sequence token identifier.
        eos_token_id: End-of-sequence token identifier.
        pretraining_tp: Tensor-parallel setting used during pretraining.
        tie_word_embeddings: Whether to tie embedding and LM head weights.
        rope_parameters: Optional RoPE configuration dictionary (mirrors HF).
        rope_scaling: Optional explicit RoPE scaling configuration.
        rope_theta: Optional rotary base if not provided in `rope_parameters`.
        attention_bias: If True, include biases in Q, K, V projections.
        attention_out_bias: Bias flag for attention output projection.
        attention_dropout: Dropout probability applied in attention weights.
        residual_dropout: Dropout applied to residual connections.
        mlp_bias: Whether to use bias terms in the MLP projections.
        head_dim: Attention head dimensionality. Defaults to hidden_size // num_attention_heads.
        gradient_checkpointing: Gradient checkpointing strategy.
        use_scan_mlp: Enable scan implementation for the MLP.
        scan_mlp_chunk_size: Chunk size used by scan MLP.
        bits: Optional quantisation setting.
        layer_types: Optional per-layer attention type metadata.
    """

    model_type: str = "seed_oss"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 155_136,
        hidden_size: int = 4096,
        intermediate_size: int = 27_648,
        num_hidden_layers: int = 64,
        num_attention_heads: int = 80,
        num_key_value_heads: int | None = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 524_288,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, tp.Any] | None = None,
        rope_scaling: dict[str, tp.Any] | None = None,
        rope_theta: float | None = None,
        attention_bias: bool = True,
        attention_out_bias: bool = False,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        mlp_bias: bool = False,
        head_dim: int | None = 128,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        bits: int | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ) -> None:
        rope_scaling_kwarg = kwargs.pop("rope_scaling", None)
        rope_theta_kwarg = kwargs.pop("rope_theta", None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings

        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        self.attention_dropout = attention_dropout
        self.resid_pdrop = residual_dropout
        self.mlp_bias = mlp_bias

        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits

        self.layer_types = layer_types or ["full_attention" for _ in range(num_hidden_layers)]

        head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.head_dim = head_dim

        self.rope_parameters = rope_parameters

        rope_theta_value = rope_theta if rope_theta is not None else rope_theta_kwarg
        if rope_parameters is not None and rope_parameters.get("rope_theta") is not None:
            rope_theta_value = rope_parameters["rope_theta"]
        if rope_theta_value is None:
            rope_theta_value = 10_000.0
        self.rope_theta = rope_theta_value

        rope_scaling_value = rope_scaling if rope_scaling is not None else rope_scaling_kwarg
        rope_scaling_from_params = _extract_rope_scaling(rope_parameters)
        if rope_scaling_from_params is not None:
            rope_scaling_value = rope_scaling_from_params
        self.rope_scaling = rope_scaling_value

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_scan_mlp=use_scan_mlp,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs):
        """Partition rules optimised for Seed-Oss tensor-parallel layouts."""
        pmgr = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmgr.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmgr.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmgr.resolve(RowWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/bias", pmgr.resolve(Replicated)),
            (r"self_attn/o_proj/bias", pmgr.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmgr.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmgr.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmgr.resolve(Replicated)),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/kernel",
                pmgr.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmgr.resolve(ColumnWise)),
            (r"score/kernel", pmgr.resolve(RowWise)),
            (r".*bias", pmgr.resolve(Replicated)),
            (r".*", pmgr.resolve(Replicated)),
        )

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Map each layer to the corresponding attention mask details."""
        mask_details: dict[int, AttnMaskDetail] = {}
        for idx, layer_type in enumerate(self.layer_types):
            mask_details[idx] = AttnMaskDetail(
                mask_type=AttnMaskType.from_hf(layer_type),
                size=None,
            )
        return mask_details


__all__ = ["SeedOssConfig"]