# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional
from functools import partial

from torch import nn

from torchtune.models.llama3._component_builders import llama3, lora_llama3
from torchtune.models.deepseek_lite._component_builders import deepseek

from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import TikTokenTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES


def deepseek_small() -> TransformerDecoder:
    """
    Builder for creating a Llama3 model initialized w/ the default 8b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3 8B model
    """
    return deepseek(
        vocab_size=32_000,
        num_layers=12,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=768,
        max_seq_len=2048,
        intermediate_dim=3096,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,

        moe_intermediate_size=384,
        n_experts=32,
        num_experts_per_tok=6,
        aux_loss_alpha=0.001,
    )
