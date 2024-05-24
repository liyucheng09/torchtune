# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Literal, Optional

from torch import nn

from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp

from torchtune.modules import (
    CausalSelfAttention,
    FeedForward,
    KVCache,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerDecoderLayer,
    DeepSeekMoE
)

from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

from torchtune.modules.peft import LORA_ATTN_MODULES, LoRALinear

"""
Component builders for the Llama3 model and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``CausalSelfAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


# ------------------ Vanilla Llama3 ------------------


def deepseek(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,

    moe_intermediate_size = 384,
    n_experts = 32,
    num_experts_per_tok = 6,
    aux_loss_alpha = 0.001,
) -> TransformerDecoder:

    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    
    moe = DeepSeekMoE(
        dim=embed_dim,
        moe_dim=moe_intermediate_size,
        num_experts=n_experts,
        num_experts_per_tok=num_experts_per_tok,
        aux_loss_alpha=aux_loss_alpha,
    )

    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=moe,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )