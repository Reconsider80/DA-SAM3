"""
Patch SAM3 fusion encoder FFN blocks with Dual-Adaptive MoE layers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from da_sam3.models.da_moe import DAMoEConfig, DAMoEFFNWrapper


class MoEContextStore:
    concept_token: Optional[torch.Tensor] = None
    visual_memory: Optional[torch.Tensor] = None
    concept_emb: Optional[torch.Tensor] = None

    @classmethod
    def set(
        cls,
        concept_token: torch.Tensor,
        visual_memory: torch.Tensor,
        concept_emb: torch.Tensor,
    ) -> None:
        cls.concept_token = concept_token
        cls.visual_memory = visual_memory
        cls.concept_emb = concept_emb

    @classmethod
    def clear(cls) -> None:
        cls.concept_token = None
        cls.visual_memory = None
        cls.concept_emb = None


def hierarchical_moe_layer_indices(num_layers: int) -> List[int]:
    depths = [
        max(1, num_layers // 6),
        max(1, num_layers // 4),
        max(1, num_layers // 2),
    ]
    return sorted({d - 1 for d in depths if 0 <= d - 1 < num_layers})


def _make_da_moe_forward_pre(layer: nn.Module, da_moe_ffn: DAMoEFFNWrapper):
    def forward_pre(
        tgt,
        memory,
        dac: bool = False,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if dac:
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]

        tgt2 = layer.norm1(tgt)
        q = k = tgt2 + query_pos if layer.pos_enc_at_attn else tgt2
        tgt2 = layer.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + layer.dropout1(tgt2)
        if dac:
            tgt = torch.cat((tgt, other_tgt), dim=0)

        tgt2 = layer.norm2(tgt)
        tgt2 = layer.cross_attn_image(
            query=tgt2 + query_pos if layer.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if layer.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + layer.dropout2(tgt2)

        ctx = MoEContextStore
        if ctx.concept_token is not None:
            tgt = da_moe_ffn(
                tgt,
                ctx.concept_token,
                ctx.visual_memory,
                ctx.concept_emb,
                pre_norm=True,
            )
        else:
            tgt2 = layer.norm3(tgt)
            tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt2))))
            tgt = tgt + layer.dropout3(tgt2)
        return tgt

    return forward_pre


def _make_da_moe_forward_post(layer: nn.Module, da_moe_ffn: DAMoEFFNWrapper):
    def forward_post(
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        **kwargs,
    ):
        q = k = tgt + query_pos if layer.pos_enc_at_attn else tgt
        tgt2 = layer.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + layer.dropout1(tgt2)
        tgt = layer.norm1(tgt)

        tgt2 = layer.cross_attn_image(
            query=tgt + query_pos if layer.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if layer.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + layer.dropout2(tgt2)
        tgt = layer.norm2(tgt)

        ctx = MoEContextStore
        if ctx.concept_token is not None:
            tgt = da_moe_ffn(
                tgt,
                ctx.concept_token,
                ctx.visual_memory,
                ctx.concept_emb,
                pre_norm=False,
            )
        else:
            tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt))))
            tgt = tgt + layer.dropout3(tgt2)
            tgt = layer.norm3(tgt)
        return tgt

    return forward_post


def inject_da_moe_into_sam3(
    model: nn.Module,
    config: DAMoEConfig,
) -> Tuple[nn.Module, List[int], Dict[int, DAMoEFFNWrapper]]:
    encoder = model.transformer.encoder
    num_layers = len(encoder.layers)
    moe_indices = hierarchical_moe_layer_indices(num_layers)
    wrappers: Dict[int, DAMoEFFNWrapper] = {}

    for idx in moe_indices:
        layer = encoder.layers[idx]
        da_moe_ffn = DAMoEFFNWrapper(
            linear1=layer.linear1,
            linear2=layer.linear2,
            norm3=layer.norm3,
            dropout3=layer.dropout3,
            config=config,
            activation=layer.activation,
        )
        if layer.pre_norm:
            layer.forward_pre = _make_da_moe_forward_pre(layer, da_moe_ffn)
        else:
            layer.forward_post = _make_da_moe_forward_post(layer, da_moe_ffn)
        layer._da_moe_ffn = da_moe_ffn  # noqa: SLF001
        wrappers[idx] = da_moe_ffn

    return model, moe_indices, wrappers


def collect_moe_aux_losses(model: nn.Module) -> Dict[str, torch.Tensor]:
    balance_terms, sparse_terms = [], []
    for layer in model.transformer.encoder.layers:
        wrapper = getattr(layer, "_da_moe_ffn", None)
        if wrapper is None:
            continue
        aux = wrapper.aux_losses()
        balance_terms.append(aux["balance"])
        sparse_terms.append(aux["sparse"])

    device = next(model.parameters()).device
    if not balance_terms:
        zero = torch.tensor(0.0, device=device)
        return {"balance": zero, "sparse": zero}
    return {
        "balance": torch.stack(balance_terms).mean(),
        "sparse": torch.stack(sparse_terms).mean(),
    }


def configure_trainable_parameters(model: nn.Module, stage: str = "warmup") -> None:
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.transformer.encoder.layers:
        for norm in (getattr(layer, "norm1", None), getattr(layer, "norm2", None), getattr(layer, "norm3", None)):
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = True

        wrapper = getattr(layer, "_da_moe_ffn", None)
        if wrapper is None:
            continue

        train_experts = stage == "warmup"
        for p in wrapper.da_moe.experts.parameters():
            p.requires_grad = train_experts
        for p in wrapper.da_moe.router.parameters():
            p.requires_grad = True
        for p in wrapper.da_moe.domain_context.parameters():
            p.requires_grad = True
