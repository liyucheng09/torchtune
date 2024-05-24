import copy
from typing import Optional

import torch
from torch import nn, Tensor

from torchtune.modules import CausalSelfAttention, KVCache, FeedForward
import numpy as np

class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class DeepseekMLP(nn.Module):
    def __init__(
        self,
        dim,
        moe_dim
    ):
        super().__init__()
        
        self.gate_proj = nn.Linear(dim, moe_dim, bias=False)
        self.up_proj = nn.Linear(dim, moe_dim, bias=False)
        self.down_proj = nn.Linear(moe_dim, dim, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class MoEGate(nn.Module):
    def __init__(
        self,
        dim,
        num_experts_per_tok = 6,
        num_experts = 32,
        aux_loss_alpha = 0.001,
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.weight = nn.Linear(dim, num_experts, bias=False)
    
    def forward(self, x: Tensor):
        bsz, seq_len, dim = x.shape
        hidden_states = hidden_states.view(-1, dim)

        logits = self.weight(hidden_states)
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)

        topk_weight, topk_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1, sorted=False
        )

        scores = scores.view(bsz, seq_len, -1)
        topk_indices_for_aux_loss = topk_indices.view(bsz, seq_len, -1)
        ce = torch.zeros(
            bsz, self.num_experts, device=scores.device
        )
        ce.scatter_add_(
            1,
            topk_indices_for_aux_loss,
            torch.ones(bsz, seq_len * self.num_experts_per_tok, device=scores.device),
        ).div_(seq_len * self.num_experts_per_tok / self.num_experts)

        aux_loss = (ce * scores.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha

        return topk_indices, topk_weight, aux_loss
        
class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        dim,
        moe_dim,
        num_experts_per_tok,
        num_experts,
        aux_loss_alpha = 0.001,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                DeepseekMLP(
                    dim=dim,
                    moe_dim=moe_dim,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = MoEGate(
            dim=dim,
            num_experts_per_tok=num_experts_per_tok,
            num_experts=num_experts,
            aux_loss_alpha=aux_loss_alpha,
        )

    def forward(
        self,
        x,
    ):
        x_shape = x.shape
        topk_indices, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x_shape[-1])
        flat_topk_indices = topk_indices.view(-1)

        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_indices == i] = expert(hidden_states[flat_topk_indices == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*x_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_inference(hidden_states, topk_indices, topk_weight).view(*x_shape)
        return y
    
    @torch.no_grad()
    def moe_inference(
        self,
        x,
        topk_ids,
        topk_weight,
    ):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        tokens_per_expert_group = tokens_per_expert.new_empty(
            tokens_per_expert.shape[0]
        )
        torch.distributed.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
        output_splits = (
            tokens_per_expert_group.view(self.ep_size, -1)
            .sum(1)
            .cpu()
            .numpy()
            .tolist()
        )
        gathered_tokens = sorted_tokens.new_empty(
            tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
        )
        input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
        torch.distributed.all_to_all(
            list(gathered_tokens.split(output_splits)),
            list(sorted_tokens.split(input_split_sizes)),
        )
        tokens_per_expert_post_gather = tokens_per_expert_group.view(
            self.ep_size, self.experts_per_rank
        ).sum(dim=0)
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % self.experts_per_rank
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[gatherd_idxs] = outs
        gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
        torch.distributed.all_to_all(
            list(gathered_tokens.split(input_split_sizes)),
            list(new_x.split(output_splits)),
        )
        outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out