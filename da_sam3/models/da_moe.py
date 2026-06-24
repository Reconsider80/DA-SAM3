"""
Dual-Adaptive MoE Layer for DA-SAM3.

Implements:
  - Dynamic Expert Router (DER): multimodal sparse routing
  - Decomposed Parameterized Experts (DPE): frozen base FFN + low-rank deltas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DAMoEConfig:
    d_model: int = 256
    dim_feedforward: int = 2048
    num_experts: int = 4
    top_k: int = 2
    rank: int = 8
    dropout: float = 0.1
    activation: str = "relu"


class DomainContextCrossAttention(nn.Module):
    """Compute h_ctx = CrossAttn(C_tok, Pool(V))."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        concept_token: torch.Tensor,
        visual_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            concept_token: (B, 1, D) concept / text token
            visual_memory: (B, N, D) spatial visual tokens
        Returns:
            h_ctx: (B, D) global domain-context vector
        """
        pooled = visual_memory.mean(dim=1, keepdim=True)
        ctx, _ = self.cross_attn(concept_token, pooled, pooled)
        return ctx.squeeze(1)


class DynamicExpertRouter(nn.Module):
    """
    Token-wise sparse router conditioned on local features, global context,
    and frozen concept semantics (Eq. 2-3 in DA-SAM3 paper).
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.projection = nn.Linear(d_model * 3, num_experts, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        h_ctx: torch.Tensor,
        concept_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, D)
            h_ctx: (B, D)
            concept_emb: (B, D) frozen concept embedding
        Returns:
            routing_weights: (B, N, E) sparse softmax weights
            router_logits: (B, N, E) raw logits for aux losses
        """
        bsz, num_tokens, dim = tokens.shape
        h_ctx_exp = h_ctx.unsqueeze(1).expand(-1, num_tokens, -1)
        concept_exp = concept_emb.detach().unsqueeze(1).expand(-1, num_tokens, -1)
        router_input = torch.cat([tokens, h_ctx_exp, concept_exp], dim=-1)
        logits = self.projection(router_input)

        topk_logits, topk_indices = logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits.scatter_(-1, topk_indices, topk_logits)
        routing_weights = F.softmax(sparse_logits, dim=-1)
        return routing_weights, logits


class DecomposedParameterizedExpert(nn.Module):
    """
    Shared frozen FFN base (W0) + expert-specific low-rank delta (Eq. 4-5).
    """

    def __init__(
        self,
        base_linear1: nn.Linear,
        base_linear2: nn.Linear,
        rank: int,
        activation: nn.Module,
        dropout: float,
    ):
        super().__init__()
        self.base_linear1 = base_linear1
        self.base_linear2 = base_linear2
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        d_model = base_linear1.in_features
        d_ff = base_linear1.out_features

        self.delta_a1 = nn.Parameter(torch.zeros(d_model, rank))
        self.delta_b1 = nn.Parameter(torch.zeros(rank, d_ff))
        self.delta_a2 = nn.Parameter(torch.zeros(d_ff, rank))
        self.delta_b2 = nn.Parameter(torch.zeros(rank, d_model))

        nn.init.zeros_(self.delta_a1)
        nn.init.zeros_(self.delta_b1)
        nn.init.zeros_(self.delta_a2)
        nn.init.zeros_(self.delta_b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w1 = self.delta_a1 @ self.delta_b1
        delta_w2 = self.delta_a2 @ self.delta_b2

        w1 = self.base_linear1.weight + delta_w1.t()
        b1 = self.base_linear1.bias
        w2 = self.base_linear2.weight + delta_w2.t()
        b2 = self.base_linear2.bias

        hidden = self.activation(F.linear(x, w1, b1))
        hidden = self.dropout(hidden)
        return F.linear(hidden, w2, b2)


class DualAdaptiveMoELayer(nn.Module):
    """
    Full Dual-Adaptive MoE layer replacing a standard Transformer FFN block.
    """

    def __init__(
        self,
        linear1: nn.Linear,
        linear2: nn.Linear,
        config: DAMoEConfig,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        if activation is None:
            activation = nn.ReLU() if config.activation == "relu" else nn.GELU()

        for param in linear1.parameters():
            param.requires_grad = False
        for param in linear2.parameters():
            param.requires_grad = False

        self.domain_context = DomainContextCrossAttention(
            config.d_model, dropout=config.dropout
        )
        self.router = DynamicExpertRouter(
            config.d_model, config.num_experts, config.top_k
        )
        self.experts = nn.ModuleList(
            [
                DecomposedParameterizedExpert(
                    linear1, linear2, config.rank, activation, config.dropout
                )
                for _ in range(config.num_experts)
            ]
        )
        self.last_routing_weights: Optional[torch.Tensor] = None
        self.last_router_logits: Optional[torch.Tensor] = None

    def forward(
        self,
        tokens: torch.Tensor,
        concept_token: torch.Tensor,
        visual_memory: torch.Tensor,
        concept_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) FFN input after attention blocks
            concept_token: (B, 1, D) first text/concept token in sequence
            visual_memory: (B, N_v, D) image feature tokens
            concept_emb: (B, D) pooled concept embedding
        """
        h_ctx = self.domain_context(concept_token, visual_memory)
        routing_weights, router_logits = self.router(tokens, h_ctx, concept_emb)
        self.last_routing_weights = routing_weights
        self.last_router_logits = router_logits

        bsz, num_tokens, dim = tokens.shape
        output = torch.zeros_like(tokens)

        flat_tokens = tokens.reshape(-1, dim)
        flat_weights = routing_weights.reshape(-1, self.config.num_experts)

        for expert_idx, expert in enumerate(self.experts):
            expert_weight = flat_weights[:, expert_idx]
            active = expert_weight > 0
            if not active.any():
                continue
            expert_out = expert(flat_tokens[active])
            output.reshape(-1, dim)[active] += expert_out * expert_weight[active].unsqueeze(-1)

        return output

    def aux_losses(self) -> Dict[str, torch.Tensor]:
        """Load balancing and sparsity auxiliary losses (Eq. 7)."""
        if self.last_routing_weights is None:
            zero = torch.tensor(0.0, device=next(self.parameters()).device)
            return {"balance": zero, "sparse": zero}

        weights = self.last_routing_weights
        probs = weights.mean(dim=(0, 1))
        usage = (weights > 0).float().mean(dim=(0, 1))
        balance = self.config.num_experts * torch.sum(probs * usage)

        sparse = weights.abs().mean()
        return {"balance": balance, "sparse": sparse}


class DAMoEFFNWrapper(nn.Module):
    """
    Drop-in replacement for TransformerEncoderLayer FFN (post-norm residual path).
    """

    def __init__(
        self,
        linear1: nn.Linear,
        linear2: nn.Linear,
        norm3: nn.LayerNorm,
        dropout3: nn.Dropout,
        config: DAMoEConfig,
        activation: nn.Module,
    ):
        super().__init__()
        self.da_moe = DualAdaptiveMoELayer(linear1, linear2, config, activation)
        self.norm3 = norm3
        self.dropout3 = dropout3

    def forward(
        self,
        tgt: torch.Tensor,
        concept_token: torch.Tensor,
        visual_memory: torch.Tensor,
        concept_emb: torch.Tensor,
        pre_norm: bool = True,
    ) -> torch.Tensor:
        # tgt: (seq, batch, dim)
        x = tgt.transpose(0, 1)
        c_tok = concept_token.transpose(0, 1) if concept_token.dim() == 3 else concept_token
        v_mem = visual_memory.transpose(0, 1)
        if c_tok.dim() == 2:
            c_tok = c_tok.unsqueeze(1)

        if pre_norm:
            normed = self.norm3(tgt).transpose(0, 1)
            out = self.da_moe(normed, c_tok, v_mem, concept_emb)
            out = tgt + self.dropout3(out.transpose(0, 1))
            return out

        out = self.da_moe(x, c_tok, v_mem, concept_emb)
        out = tgt + self.dropout3(out.transpose(0, 1))
        out = self.norm3(out)
        return out

    def aux_losses(self) -> Dict[str, torch.Tensor]:
        return self.da_moe.aux_losses()
