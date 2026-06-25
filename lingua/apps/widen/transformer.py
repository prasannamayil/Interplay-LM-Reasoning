# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Widen-the-line architecture zoo (Exp 3 of RESEARCH_PROPOSAL.md).
#
# One configurable LM that realises several AR architectures/parameterisations
# behind a single `arch_type` knob, so they can be trained on identical data and
# evaluated with the identical lingua harness — the cleanest way to show that
# changing the *architecture* keeps a model on the universality line, while
# changing the *objective* (diffusion, handled elsewhere) leaves it.
#
#   arch_type:
#     "dense"        standard Llama-style transformer (RoPE + SwiGLU)  [reference]
#     "gqa"          dense, just configure a small n_kv_heads          [config only]
#     "moe"          top-k Mixture-of-Experts SwiGLU FFN
#     "looped"       weight-tied Universal/Looped Transformer (recurrent depth)
#     "sliding"      sliding-window (local) causal attention
#     "linear"       softmax-free (feature-map) causal attention
#     "tokenformer"  token-parameter (Pattention) projections + FFN
#
# Compatibility notes (see gsm_infinity/README.md):
#   * dense / gqa / moe / looped / tokenformer keep a `lingua.transformer.Attention`
#     instance per block, so the eval generator's KV-cache + packed prefill work
#     unchanged → these are *fully eval-faithful*.
#   * sliding / linear use a custom attention path; they train correctly but the
#     stock packed-generation eval path applies the standard causal mask. They are
#     marked train-validated with an eval-faithfulness TODO.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from xformers.ops import fmha, AttentionBias

from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    InitStdFactor,
    RMSNorm,
    RotaryEmbedding,
    TiedLinear,
    Attention,
    FeedForward,
    apply_rotary_emb,
    repeat_kv,
    cross_entropy,
)


# ---------------------------------------------------------------------------
# masks / flops (mirrors apps/main/transformer.py)
# ---------------------------------------------------------------------------
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "fmha":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "fmha":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    # n_layers here should be the *effective* compute depth (loops counted).
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


# ---------------------------------------------------------------------------
# MoE feed-forward
# ---------------------------------------------------------------------------
class Expert(nn.Module):
    """A single SwiGLU expert (same shape as lingua FeedForward, no bias)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self, init_std=None, factor=1.0):
        in_std = init_std or (self.dim ** (-0.5))
        out_std = (init_std or (self.hidden_dim ** (-0.5))) / factor
        for w in (self.w1, self.w3):
            nn.init.trunc_normal_(w.weight, std=in_std, a=-3 * in_std, b=3 * in_std)
        nn.init.trunc_normal_(self.w2.weight, std=out_std, a=-3 * out_std, b=3 * out_std)


def _swiglu_hidden(dim, multiple_of, ffn_dim_multiplier):
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class MoEFeedForward(nn.Module):
    """Top-k MoE over SwiGLU experts with a GShard load-balancing aux loss.

    The aux loss is stashed on ``self.aux_loss`` each forward; the LM head sums it
    across blocks and adds ``moe_aux_coef * aux`` to the cross-entropy.
    """

    def __init__(
        self,
        dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        num_experts: int,
        top_k: int,
        expert_ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        # Each expert is sized like a (possibly shrunk) dense FFN. Shrinking the
        # expert hidden by top_k keeps *active* params ~= a dense FFN.
        mult = expert_ffn_dim_multiplier
        if mult is None:
            mult = (ffn_dim_multiplier or 1.0) / top_k
        hidden_dim = _swiglu_hidden(dim, multiple_of, mult)
        self.hidden_dim = hidden_dim
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim) for _ in range(num_experts)]
        )
        self.aux_loss = torch.zeros((), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.reshape(-1, dim)  # (N, D)
        logits = self.gate(x_flat)  # (N, E)
        probs = torch.softmax(logits.float(), dim=-1).type_as(x)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # (N, k)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x_flat)
        for slot in range(self.top_k):
            idx = topk_idx[:, slot]  # (N,)
            weight = topk_probs[:, slot].unsqueeze(-1)  # (N,1)
            for e in range(self.num_experts):
                token_mask = idx == e
                if token_mask.any():
                    sel = x_flat[token_mask]
                    out[token_mask] += weight[token_mask] * self.experts[e](sel)

        # GShard load-balancing aux loss: E * sum_e f_e * P_e
        # f_e = fraction of tokens dispatched to e (top-1 for the counting), P_e = mean router prob.
        with torch.no_grad():
            top1 = topk_idx[:, 0]
            f = torch.zeros(self.num_experts, device=x.device, dtype=probs.dtype)
            f.scatter_add_(0, top1, torch.ones_like(top1, dtype=probs.dtype))
            f = f / max(1, top1.numel())
        P = probs.mean(dim=0)
        self.aux_loss = self.num_experts * torch.sum(f * P)
        return out.reshape(bsz, seqlen, dim)

    def reset_parameters(self, init_std=None, factor=1.0):
        gstd = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(self.gate.weight, std=gstd, a=-3 * gstd, b=3 * gstd)
        for e in self.experts:
            e.reset_parameters(init_std, factor)


# ---------------------------------------------------------------------------
# Linear (softmax-free) attention  [train-validated; eval TODO]
# ---------------------------------------------------------------------------
class LinearAttention(nn.Module):
    """Feature-map (elu+1) causal attention, O(L^2) masked form for simplicity.

    Not a ``lingua.transformer.Attention`` subclass on purpose: the eval generator
    only attaches a KV-cache to that type, and incremental KV-cached generation is
    not implemented here. Training (single causal stream) is correct.
    """

    def __init__(self, dim, head_dim, n_heads, n_kv_heads, rope_theta):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        bsz, seq_len, _ = x.shape
        xq = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)
        # B S H D -> B H S D
        q, k, v = (t.transpose(1, 2) for t in (xq, xk, xv))
        q, k = self._phi(q), self._phi(k)
        scores = torch.matmul(q, k.transpose(-1, -2))  # B H S S
        causal = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal, 0.0)
        denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out = torch.matmul(scores / denom, v)  # B H S D
        out = out.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        return self.wo(out)

    def reset_parameters(self, init_std=None, factor=1.0):
        std = init_std or (self.dim ** (-0.5))
        for w in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(w.weight, std=std, a=-3 * std, b=3 * std)
        ostd = std / factor
        nn.init.trunc_normal_(self.wo.weight, std=ostd, a=-3 * ostd, b=3 * ostd)


# ---------------------------------------------------------------------------
# Tokenformer (Pattention) building blocks
# ---------------------------------------------------------------------------
class PattentionLinear(nn.Module):
    """Token-parameter attention as a drop-in for nn.Linear(in_dim, out_dim).

    y = softmax(x @ K^T / sqrt(in)) @ V, with K (m,in), V (m,out) learnable
    "parameter tokens" (Wang et al., Tokenformer, 2410.23168).
    """

    def __init__(self, in_dim: int, out_dim: int, num_tokens: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        self.key = nn.Parameter(torch.empty(num_tokens, in_dim))
        self.value = nn.Parameter(torch.empty(num_tokens, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.linear(x, self.key)  # (..., m)
        a = torch.softmax(a.float() / math.sqrt(self.in_dim), dim=-1).type_as(x)
        return torch.matmul(a, self.value)  # (..., out)

    def reset_parameters(self, init_std=None, factor=1.0):
        kstd = init_std or (self.in_dim ** (-0.5))
        vstd = (init_std or (self.num_tokens ** (-0.5))) / factor
        nn.init.trunc_normal_(self.key, std=kstd, a=-3 * kstd, b=3 * kstd)
        nn.init.trunc_normal_(self.value, std=vstd, a=-3 * vstd, b=3 * vstd)


class TokenformerAttention(Attention):
    """Standard softmax token-token attention, but with Pattention projections.

    Subclasses ``Attention`` so the eval generator's KV-cache machinery works
    unchanged (it keys off isinstance(module, Attention)).
    """

    def __init__(self, dim, head_dim, n_heads, n_kv_heads, rope_theta, num_tokens):
        super().__init__(dim, head_dim, n_heads, n_kv_heads, rope_theta)
        self.wq = PattentionLinear(dim, n_heads * head_dim, num_tokens)
        self.wk = PattentionLinear(dim, n_kv_heads * head_dim, num_tokens)
        self.wv = PattentionLinear(dim, n_kv_heads * head_dim, num_tokens)
        self.wo = PattentionLinear(n_heads * head_dim, dim, num_tokens)

    def reset_parameters(self, init_std=None, factor=1.0):
        for w in (self.wq, self.wk, self.wv):
            w.reset_parameters(init_std, 1.0)
        self.wo.reset_parameters(init_std, factor)


class PattentionFeedForward(nn.Module):
    """SwiGLU FFN with Pattention projections."""

    def __init__(self, dim, multiple_of, ffn_dim_multiplier, num_tokens):
        super().__init__()
        hidden_dim = _swiglu_hidden(dim, multiple_of, ffn_dim_multiplier)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = PattentionLinear(dim, hidden_dim, num_tokens)
        self.w3 = PattentionLinear(dim, hidden_dim, num_tokens)
        self.w2 = PattentionLinear(hidden_dim, dim, num_tokens)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self, init_std=None, factor=1.0):
        self.w1.reset_parameters(init_std, 1.0)
        self.w3.reset_parameters(init_std, 1.0)
        self.w2.reset_parameters(init_std, factor)


# ---------------------------------------------------------------------------
# Block + model
# ---------------------------------------------------------------------------
@dataclass
class LMWidenArgs(BaseTransformerArgs):
    seed: int = 42
    vocab_size: int = -1
    weight_tying: bool = False

    attn_impl: str = "sdpa"
    sliding_window: Optional[int] = None

    # which architecture this run realises
    arch_type: str = "dense"  # dense|gqa|moe|looped|sliding|linear|tokenformer

    # MoE
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_aux_coef: float = 0.01
    moe_expert_ffn_dim_multiplier: Optional[float] = None

    # Looped / Universal transformer
    looped_n_unique_layers: int = 6
    looped_n_loops: int = 4

    # Tokenformer
    pattention_num_tokens: int = 1024


def _build_attention(args: LMWidenArgs, head_dim, n_heads, n_kv_heads):
    if args.arch_type == "linear":
        return LinearAttention(args.dim, head_dim, n_heads, n_kv_heads, args.rope_theta)
    if args.arch_type == "tokenformer":
        return TokenformerAttention(
            args.dim, head_dim, n_heads, n_kv_heads, args.rope_theta,
            args.pattention_num_tokens,
        )
    return Attention(args.dim, head_dim, n_heads, n_kv_heads, args.rope_theta)


def _build_ffn(args: LMWidenArgs):
    if args.arch_type == "moe":
        return MoEFeedForward(
            dim=args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            num_experts=args.moe_num_experts,
            top_k=args.moe_top_k,
            expert_ffn_dim_multiplier=args.moe_expert_ffn_dim_multiplier,
        )
    if args.arch_type == "tokenformer":
        return PattentionFeedForward(
            args.dim, args.multiple_of, args.ffn_dim_multiplier,
            args.pattention_num_tokens,
        )
    return FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
    )


class WidenBlock(nn.Module):
    def __init__(self, args: LMWidenArgs):
        super().__init__()
        head_dim = args.head_dim or args.dim // args.n_heads
        n_heads = args.n_heads or args.dim // args.head_dim
        n_kv_heads = args.n_kv_heads or n_heads
        assert n_heads % n_kv_heads == 0
        assert args.dim % n_heads == 0
        self.attention = _build_attention(args, head_dim, n_heads, n_kv_heads)
        self.feed_forward = _build_ffn(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        h = x + self.attention(
            self.attention_norm(x), freq_cis, tok_idx=tok_idx, mask=mask,
            attn_impl=attn_impl,
        )
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class WidenTransformer(BaseTransformer):
    """BaseTransformer with a configurable block type and optional depth looping.

    init_weights() is inherited (iterates self.layers); we only override layer
    construction and the forward loop.
    """

    def __init__(self, args: LMWidenArgs):
        nn.Module.__init__(self)
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
        )
        self.arch_type = args.arch_type
        self.n_loops = args.looped_n_loops if args.arch_type == "looped" else 1
        n_unique = (
            args.looped_n_unique_layers if args.arch_type == "looped" else args.n_layers
        )
        self.layers = nn.ModuleList([WidenBlock(args) for _ in range(n_unique)])

    def forward(self, h, tok_idx=None, mask=None, attn_impl="sdpa"):
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)
        for _ in range(self.n_loops):
            for layer in self.layers:
                h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h


class LMWiden(WidenTransformer):
    def __init__(self, args: LMWidenArgs):
        super().__init__(args)
        assert args.vocab_size > 0
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        self.attn_impl = args.attn_impl
        self.moe_aux_coef = args.moe_aux_coef

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def _sliding_bool_mask(self, seqlen, device):
        i = torch.arange(seqlen, device=device)
        causal = i.unsqueeze(1) >= i.unsqueeze(0)
        window = (i.unsqueeze(1) - i.unsqueeze(0)) < self.sliding_window
        return (causal & window).view(1, 1, seqlen, seqlen)

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask=None,
        attn_impl: str = "sdpa",
    ):
        bsz, seqlen = token_values.shape
        h = self.tok_embeddings(token_values)

        if mask is None:
            if self.arch_type == "sliding" and self.sliding_window and attn_impl == "sdpa":
                mask = self._sliding_bool_mask(seqlen, token_values.device)
            else:
                mask = create_causal_mask(seqlen, attn_impl, self.sliding_window)

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        logits = self.output(self.norm(h))

        if target is not None:
            loss = cross_entropy(logits, target)
            if self.arch_type == "moe":
                aux = sum(
                    m.aux_loss
                    for m in self.modules()
                    if isinstance(m, MoEFeedForward)
                )
                loss = loss + self.moe_aux_coef * aux
            return loss
        return logits

    def reset_parameters(self, init_std=None):
        super().reset_parameters()  # rope
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight, std=init_std, a=-3 * init_std, b=3 * init_std
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight, std=init_std, a=-3 * init_std, b=3 * init_std
            )


# ---------------------------------------------------------------------------
# fsdp / tp hooks (parity with apps/main/transformer.py)
# ---------------------------------------------------------------------------
def get_no_recompute_ops():
    return None


def build_fsdp_grouping_plan(model_args: LMWidenArgs) -> List[Tuple[str, bool]]:
    group_plan: List[Tuple[str, bool]] = []
    group_plan.append(("tok_embeddings", False))
    n_unique = (
        model_args.looped_n_unique_layers
        if model_args.arch_type == "looped"
        else model_args.n_layers
    )
    for i in range(n_unique):
        group_plan.append((f"layers.{i}", False))
    group_plan.append(("output", True))
    return group_plan


def tp_parallelize(model, tp_mesh, model_args: LMWidenArgs, distributed_args):
    raise NotImplementedError(
        "Tensor parallelism is not implemented for apps.widen; use tp_size=1."
    )
