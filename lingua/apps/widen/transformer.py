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
#     "parallel"     PaLM/GPT-J parallel block: attn + FFN from the SAME normed input
#     "diff"         Differential Transformer (two softmax maps subtracted, learnable λ)
#     "mamba2"       Mamba-2 selective state-space mixer (+ SwiGLU MLP, Jamba/Samba-style)
#     "fastrnn"      minGRU gated-RNN mixer via parallel scan (+ SwiGLU MLP)
#     "gla"          gated linear attention (data-dependent scalar decay + output gate)
#     "mla"          Multi-head Latent Attention (low-rank KV + decoupled RoPE)
#
# Compatibility notes (see gsm_infinity/README.md):
#   * dense / gqa / moe / looped / tokenformer / parallel / diff keep a
#     `lingua.transformer.Attention` instance per block, so the eval generator's
#     KV-cache + packed prefill work unchanged → these are *fully eval-faithful*.
#   * mla uses standard softmax attention that respects the mask, so its *cloze /
#     loglikelihood* eval (FineWeb) is faithful; only incremental KV-cached
#     generation (GSM pass@k) needs a custom latent cache (eval-TODO).
#   * sliding / linear / gla / mamba2 / fastrnn use a custom mixer path; they train
#     correctly but the stock packed-generation eval path applies the standard
#     causal mask and the recurrent mixers ignore the packed-doc mask (cross-doc
#     state leakage). Marked train-validated with an eval-faithfulness TODO.
#   * mamba2 / fastrnn lazily import the proven kernels from apps/mamba and
#     apps/fastRNN (mamba_ssm, causal_conv1d, accelerated_scan). The import is
#     deferred to construction time so the rest of the zoo loads even when those
#     CUDA/Triton packages are absent.

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
    flex_attention_comp,
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
        # Native-dtype softmax (NOT .float()): under FSDP mixed precision (bf16 params,
        # fp32 reduce), an fp32 intermediate in the router leaks an fp32 gradient onto the
        # gate, so the MoE block ends up with mixed {fp32,bf16} grads and FSDP's
        # reduce-scatter asserts "uniform gradient dtype". Keeping the router in the param
        # dtype keeps every MoE grad bf16. (1-GPU GSM never reduce-scatters, so this only
        # bit the 8-GPU FineWeb MoE run.)
        probs = torch.softmax(logits, dim=-1)
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

        # Every expert MUST get a gradient each step. With top-k routing an expert can
        # receive ZERO tokens in a microbatch (likely with many MoE layers) -> its params
        # get no grad -> FSDP fills the missing grad in fp32 -> mixed-dtype reduce-scatter
        # assert ("uniform gradient dtype {float32, bfloat16}"). A zero-magnitude touch of
        # all expert params guarantees each gets a (0) grad in the param dtype (bf16).
        touch = sum(p.sum() for e in self.experts for p in e.parameters())
        out = out + 0.0 * touch

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

    y = Θ(x @ K^T) @ V, with K (m,in), V (m,out) learnable "parameter tokens"
    (Wang et al., Tokenformer, 2410.23168). Θ is the paper's "gelu_l2_norm"
    normalization (NOT softmax): GeLU, then L2-normalize over the parameter-token
    axis, rescaled by sqrt(m). The earlier softmax form forced every output to be a
    convex (all-positive, sum-to-one) combination of value tokens — far too
    restrictive, gradient-starved, and the model would not learn (train loss stuck
    ~1.34). gelu_l2_norm allows signed, unbounded combinations like a real linear
    map; the sqrt(m) rescale (paired with value init std m^-0.5) keeps the output
    variance ~= a standard nn.Linear so magnitudes are right at init.
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
        a = F.gelu(a.float())
        a = a / a.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        a = (a * math.sqrt(self.num_tokens)).type_as(x)
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
# shared attention dispatch (BSHD in/out; supports differing value head_dim)
# ---------------------------------------------------------------------------
def _attend(xq, xk, xv, mask, attn_impl):
    """Softmax attention over (B, S, H, D) tensors, returning (B, S, H, Dv).

    Mirrors the dispatch in lingua.transformer.Attention so custom archs (diff,
    mla) stay faithful under every eval path (sdpa training/decode, flex_attention
    packed prefill, fmha). xv may have a different last dim than xq/xk; the scale
    defaults to 1/sqrt(xq.head_dim) exactly as the stock attention does."""
    if attn_impl == "flex_attention":
        assert mask is None or isinstance(mask, BlockMask)
        q, k, v = (e.transpose(1, 2) for e in (xq, xk, xv))
        out = flex_attention_comp(q, k, v, block_mask=mask)
        return out.transpose(1, 2).contiguous()
    elif attn_impl == "fmha":
        assert mask is None or isinstance(mask, AttentionBias)
        return fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
    elif attn_impl == "sdpa":
        q, k, v = (e.transpose(1, 2) for e in (xq, xk, xv))
        assert mask is None or isinstance(mask, (str, torch.Tensor))
        is_causal = (mask == "causal") if isinstance(mask, str) else False
        m = mask if isinstance(mask, torch.Tensor) else None
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=m)
        return out.transpose(1, 2).contiguous()
    raise NotImplementedError(f"Attention implementation {attn_impl} not supported")


# ---------------------------------------------------------------------------
# Differential Transformer  [eval-faithful: subclasses Attention -> stock KVCache]
# ---------------------------------------------------------------------------
class DiffAttention(Attention):
    """Differential attention (Ye et al., 2410.05258).

    Each head carries two query/key groups; the attention map is
    softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T), which cancels common-mode
    attention noise. lambda is reparameterised from four learnable vectors so it is
    positive and ~lambda_init at start. Head outputs are RMSNorm'd (GroupNorm) and
    rescaled by (1 - lambda_init).

    To keep the stock KV-cache eval path working we subclass Attention and present
    HALVED head counts with DOUBLED head_dim (so the projection weights have the
    exact same shapes as a dense Attention and the generator caches (k,v) of the
    right size). The two softmaxes are computed as two ``_attend`` calls over the
    same value tensor and subtracted -- algebraically identical to subtracting the
    two attention-weight matrices, and faithful under sdpa/flex/fmha."""

    def __init__(self, dim, head_dim, n_heads, n_kv_heads, rope_theta, lambda_init):
        assert n_heads % 2 == 0 and n_kv_heads % 2 == 0, (
            "diff attention halves the head count; n_heads and n_kv_heads must be even"
        )
        # half as many heads, each twice as wide -> identical wq/wk/wv/wo shapes.
        super().__init__(dim, 2 * head_dim, n_heads // 2, n_kv_heads // 2, rope_theta)
        self.base_head_dim = head_dim
        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.empty(head_dim))
        self.lambda_k1 = nn.Parameter(torch.empty(head_dim))
        self.lambda_q2 = nn.Parameter(torch.empty(head_dim))
        self.lambda_k2 = nn.Parameter(torch.empty(head_dim))
        # per-head sublayer norm over the (2 * base_head_dim) value channels.
        self.subln = RMSNorm(2 * head_dim, eps=1e-5)

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        bsz, seq_len, _ = x.shape
        d = self.base_head_dim
        hq, hkv = self.n_heads, self.n_kv_heads  # already halved

        xq = self.wq(x).view(bsz, seq_len, hq, 2 * d)
        xk = self.wk(x).view(bsz, seq_len, hkv, 2 * d)
        xv = self.wv(x).view(bsz, seq_len, hkv, 2 * d)

        q1, q2 = xq.split(d, dim=-1)
        k1, k2 = xk.split(d, dim=-1)
        # RoPE each query/key group at the base head_dim (matches rope_embeddings).
        q1, k1 = apply_rotary_emb(q1, k1, 1, freq_cis[0:seq_len])
        q2, k2 = apply_rotary_emb(q2, k2, 1, freq_cis[0:seq_len])

        # Cache stores the recombined (k1|k2) and v at width 2*d (head_dim of super).
        xk = torch.cat([k1, k2], dim=-1)
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)
        k1, k2 = xk.split(d, dim=-1)

        rep = self.heads_per_group
        k1, k2 = repeat_kv(k1, rep, dim=2), repeat_kv(k2, rep, dim=2)
        v = repeat_kv(xv, rep, dim=2)

        o1 = _attend(q1, k1, v, mask, attn_impl)  # (B, S, hq, 2d)
        o2 = _attend(q2, k2, v, mask, attn_impl)

        lam = (
            torch.exp(torch.dot(self.lambda_q1.float(), self.lambda_k1.float()))
            - torch.exp(torch.dot(self.lambda_q2.float(), self.lambda_k2.float()))
            + self.lambda_init
        ).type_as(o1)
        out = o1 - lam * o2
        out = self.subln(out) * (1.0 - self.lambda_init)
        out = out.reshape(bsz, seq_len, hq * 2 * d)
        return self.wo(out)

    def reset_parameters(self, init_std=None, factor=1.0):
        super().reset_parameters(init_std, factor)
        for p in (self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2):
            nn.init.normal_(p, mean=0.0, std=0.1)
        self.subln.reset_parameters()


# ---------------------------------------------------------------------------
# Gated Linear Attention  [train-validated; eval TODO]
# ---------------------------------------------------------------------------
class GatedLinearAttention(nn.Module):
    """Gated linear attention with a data-dependent per-head scalar decay.

    state_t = g_t * state_{t-1} + k_t^T v_t,  g_t in (0,1) per head, o_t = q_t state_t,
    normalised by the running key mass (linear-attention denominator) so the output is
    a convex combination of values -> bounded, no gradient blow-up. This mirrors the
    proven-stable ``LinearAttention`` (positive elu+1 features + denominator) plus the
    GLA decay gate and a SiLU output gate (Yang et al. 2312.06635).

    Computed with the standard CHUNKED recurrence so memory is O(B*H*C^2 + B*H*D^2)
    rather than O(B*H*S^2): a dense S x S form materialises a per-layer attention
    matrix that (being a matmul output) selective-AC keeps -> OOM at seq 2048. Within a
    chunk the pairwise decay is exp(logb_t - logb_s), s<=t (exponent <= 0, no overflow);
    across chunks an fp32 (D_k x D_v) numerator state and a (D_k) denominator state are
    carried. Custom causal path -> not packed-prefill faithful (eval TODO, like linear).

    An earlier un-normalised variant (raw scores @ v + per-head RMSNorm) diverged
    (grad ~1e6, loss stuck at random); the denominator normalisation fixes it."""

    def __init__(self, dim, head_dim, n_heads, n_kv_heads, rope_theta,
                 feature="elu", chunk_size=128):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = n_heads // n_kv_heads
        self.feature = feature
        self.chunk_size = chunk_size
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wg = nn.Linear(dim, n_heads, bias=True)  # per-head scalar forget gate
        self.wr = nn.Linear(dim, n_heads * head_dim, bias=False)  # output gate
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def _phi(self, x):
        # positive feature map keeps the denominator strictly positive (stable).
        return F.elu(x) + 1.0 if self.feature == "elu" else (x * x + 1.0)

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        bsz, seq_len, _ = x.shape
        xq = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        q = self._phi(xq).transpose(1, 2).float()  # B H S D (fp32 scan for stability)
        k = self._phi(xk).transpose(1, 2).float()
        v = xv.transpose(1, 2).float()
        logg = F.logsigmoid(self.wg(x).float()).transpose(1, 2)  # B H S

        C = self.chunk_size
        pad = (C - seq_len % C) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            # Padded rows are the LAST positions of the last chunk: their query outputs
            # are sliced off, and causal masking + chunk ordering means no kept query
            # attends them and no later chunk reads their state -> they cannot affect
            # any kept output regardless of the pad gate value.
            logg = F.pad(logg, (0, pad))
        nc = (seq_len + pad) // C
        B, H, D = bsz, self.n_heads, self.head_dim
        Dv = v.shape[-1]
        q = q.view(B, H, nc, C, D)
        k = k.view(B, H, nc, C, D)
        v = v.view(B, H, nc, C, Dv)
        logb = logg.view(B, H, nc, C).cumsum(dim=-1)  # inclusive cumlog within chunk

        causal = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
        eps = 1e-6
        outs = []
        S = torch.zeros(B, H, D, Dv, device=x.device, dtype=torch.float32)  # numerator state
        z = torch.zeros(B, H, D, device=x.device, dtype=torch.float32)      # denominator state
        for c in range(nc):
            qc, kc, vc = q[:, :, c], k[:, :, c], v[:, :, c]  # B H C D / Dv
            lb = logb[:, :, c]  # B H C
            # intra-chunk (positive scores * positive decay -> positive weights)
            dec = (lb.unsqueeze(-1) - lb.unsqueeze(-2)).masked_fill(~causal, float("-inf"))
            dec = torch.exp(dec)  # B H C C, 0 above diagonal, in (0,1] below
            A = torch.matmul(qc, kc.transpose(-1, -2)) * dec  # B H C C, >= 0
            num = torch.matmul(A, vc)               # B H C Dv
            den = A.sum(dim=-1)                      # B H C
            # inter-chunk: decayed read of the carried (numerator, denominator) state
            b = torch.exp(lb)                        # B H C, in (0,1]
            num = num + b.unsqueeze(-1) * torch.matmul(qc, S)            # + b*(q@S)
            den = den + b * torch.matmul(qc, z.unsqueeze(-1)).squeeze(-1)  # + b*(q.z)
            outs.append(num / (den.unsqueeze(-1) + eps))
            # state update: carry <- b_last * carry + sum_j (b_last/b_j) k_j (x v_j)
            w = torch.exp(lb[:, :, -1:] - lb).unsqueeze(-1)  # B H C 1
            kw = kc * w                                      # B H C D
            blast = torch.exp(lb[:, :, -1])[..., None, None]  # B H 1 1
            S = blast * S + torch.matmul(kw.transpose(-1, -2), vc)  # B H D Dv
            z = blast.squeeze(-1) * z + kw.sum(dim=2)               # B H D

        out = torch.cat(outs, dim=2)[:, :, :seq_len].type_as(x)  # B H S Dv
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)      # B S (H Dv)
        out = out * F.silu(self.wr(x))                           # SiLU output gate
        return self.wo(out)

    def reset_parameters(self, init_std=None, factor=1.0):
        std = init_std or (self.dim ** (-0.5))
        ostd = (init_std or ((self.n_heads * self.head_dim) ** (-0.5))) / factor
        for w in (self.wq, self.wk, self.wv, self.wr):
            nn.init.trunc_normal_(w.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.wo.weight, std=ostd, a=-3 * ostd, b=3 * ostd)
        nn.init.trunc_normal_(self.wg.weight, std=std, a=-3 * std, b=3 * std)
        # Bias the initial forget gate near 1 (sigmoid(3) ~ 0.95) for long memory.
        nn.init.constant_(self.wg.bias, 3.0)


# ---------------------------------------------------------------------------
# Multi-head Latent Attention  [cloze-faithful; KV-cached decode TODO]
# ---------------------------------------------------------------------------
class MLAttention(nn.Module):
    """DeepSeek-style Multi-head Latent Attention (low-rank KV + decoupled RoPE).

    Keys/values are reconstructed from a small per-token latent (kv_lora_rank); a
    separate RoPE key (qk_rope_head_dim, shared across heads) is concatenated with
    the per-head content (nope) key so position information survives the low-rank
    compression. Queries are likewise low-rank (q_lora_rank). After reconstruction
    this is standard softmax attention, so it respects the packing mask and is
    *cloze/loglikelihood faithful*. Incremental decode would need a latent KV cache
    (not the stock per-head KVCache) -> eval-TODO for generation only."""

    def __init__(
        self, dim, head_dim, n_heads, rope_theta,
        kv_lora_rank, q_lora_rank, qk_rope_head_dim,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.qk_nope_head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = head_dim
        self.q_head_dim = head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank

        self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank, eps=1e-5)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * self.q_head_dim, bias=False)

        # compressed kv latent + shared decoupled-RoPE key
        self.wkv_a = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, eps=1e-5)
        self.wkv_b = nn.Linear(
            kv_lora_rank, n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.wo = nn.Linear(n_heads * self.v_head_dim, dim, bias=False)

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        bsz, seq_len, _ = x.shape
        h, nope, rope, vd = (
            self.n_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        )

        q = self.wq_b(self.q_norm(self.wq_a(x))).view(bsz, seq_len, h, self.q_head_dim)
        q_nope, q_rope = q.split([nope, rope], dim=-1)

        kv = self.wkv_a(x)
        c_kv, k_rope = kv.split([self.kv_lora_rank, rope], dim=-1)
        k_rope = k_rope.view(bsz, seq_len, 1, rope)
        kv = self.wkv_b(self.kv_norm(c_kv)).view(bsz, seq_len, h, nope + vd)
        k_nope, v = kv.split([nope, vd], dim=-1)

        # decoupled RoPE on the rope sub-vectors (rope dim == base head_dim).
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, 1, freq_cis[0:seq_len])

        q = torch.cat([q_nope, q_rope], dim=-1)  # (B,S,H,q_head_dim)
        k = torch.cat([k_nope, k_rope.expand(bsz, seq_len, h, rope)], dim=-1)

        out = _attend(q, k, v, mask, attn_impl)  # (B,S,H,vd), scale=1/sqrt(q_head_dim)
        return self.wo(out.reshape(bsz, seq_len, h * vd))

    def reset_parameters(self, init_std=None, factor=1.0):
        def tn(w, fan_in, fac=1.0):
            std = (init_std or (fan_in ** (-0.5))) / fac
            nn.init.trunc_normal_(w.weight, std=std, a=-3 * std, b=3 * std)
        tn(self.wq_a, self.dim)
        tn(self.wq_b, self.q_lora_rank)
        tn(self.wkv_a, self.dim)
        tn(self.wkv_b, self.kv_lora_rank)
        tn(self.wo, self.n_heads * self.v_head_dim, factor)
        self.q_norm.reset_parameters()
        self.kv_norm.reset_parameters()


# ---------------------------------------------------------------------------
# Mamba-2 / fastRNN mixers  [adapters over the proven apps/* kernels; eval TODO]
# ---------------------------------------------------------------------------
class MambaMixer(nn.Module):
    """Mamba-2 selective state-space mixer in the attention slot.

    Thin adapter over the validated ``apps.mamba.core_mamba.SSM`` (the same fused
    mamba_ssm / causal_conv1d kernels that train apps/mamba at this scale). The SSM
    is lazily imported at construction so the rest of the zoo loads even without
    those CUDA/Triton packages. Block keeps the SwiGLU MLP -> hybrid SSM+MLP block
    (Jamba/Samba style). Training is faithful; the mixer ignores the packing mask,
    so cloze-prefill leaks state across packed docs (eval-TODO, like linear)."""

    def __init__(self, args):
        super().__init__()
        from apps.mamba.core_mamba import SSM, InitArgs

        self._init_args = InitArgs(
            A_init_min=args.mamba_a_init_min, A_init_max=args.mamba_a_init_max
        )
        self.ssm = SSM(
            dim=args.dim,
            hidden_dim=3 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            state_dim=args.mamba_state_dim,
            n_heads=args.mamba_n_heads or args.n_heads,
            n_groups=args.mamba_n_groups,
            conv_size=args.mamba_conv_size,
            dt_bias=args.mamba_dt_bias,
            D_has_head_dim=args.mamba_d_has_head_dim,
            learnable_init_states=False,
            chunk_size=args.mamba_chunk_size,
        )

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        # RoPE/mask/attn_impl are irrelevant to the SSM; full-sequence training scan.
        return self.ssm(x, tok_idx=None, cu_seqlens=None, ssm_impl="ssm")

    def reset_parameters(self, init_std=None, factor=1.0):
        # apps/mamba inits A_log/dt_bias/D with raw in-place ops (.uniform_/.log_/.fill_),
        # which apps/mamba guards via @torch.inference_mode() on init_weights. The widen
        # init path is not under no-grad, so guard here to avoid the "leaf Variable that
        # requires grad used in an in-place operation" error.
        with torch.no_grad():
            self.ssm.reset_parameters(init_std, factor, self._init_args)


class MinGRUMixer(nn.Module):
    """minGRU gated-RNN mixer (parallel-scan recurrence) in the attention slot.

    Thin adapter over the validated ``apps.fastRNN.minGRU.core_gru.GRU`` (uses the
    installed accelerated_scan + causal_conv1d kernels). Lazily imported. Hybrid
    RNN+MLP block. Same eval caveat as MambaMixer (custom causal scan, not packed-
    prefill faithful)."""

    def __init__(self, args):
        super().__init__()
        from apps.fastRNN.minGRU.core_gru import GRU

        self.gru = GRU(
            dim=args.dim,
            hidden_dim=3 * args.dim,
            n_heads=args.rnn_n_heads or args.n_heads,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            conv_size=args.rnn_conv_size,
        )

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        return self.gru(x, tok_idx=None, cu_seqlens=None, impl="parallel")

    def reset_parameters(self, init_std=None, factor=1.0):
        # GRU uses nn.init (no-grad-safe), but guard anyway for parity with MambaMixer.
        with torch.no_grad():
            self.gru.reset_parameters(init_std, factor)


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

    # Differential Transformer
    diff_lambda_init: float = 0.8

    # Mamba-2 (selective state-space) mixer. Defaults mirror the proven
    # apps/mamba/configs_fineweb recipe (state_dim 128, conv_size 4, n_groups 1).
    mamba_state_dim: int = 128
    mamba_n_groups: int = 1
    mamba_conv_size: int = 4
    mamba_n_heads: Optional[int] = None  # None -> reuse n_heads
    mamba_chunk_size: int = 256
    mamba_dt_bias: bool = False
    mamba_d_has_head_dim: bool = False
    mamba_a_init_min: float = 0.01
    mamba_a_init_max: float = 2.0

    # fastRNN (minGRU) mixer
    rnn_n_heads: Optional[int] = None  # None -> reuse n_heads
    rnn_conv_size: Optional[int] = 4

    # Gated Linear Attention (positive feature map keeps the denominator stable)
    gla_feature: str = "elu"  # elu (elu+1) | sq (x^2+1)

    # Multi-head Latent Attention (DeepSeek-style); None -> sensible defaults at build
    mla_kv_lora_rank: Optional[int] = None
    mla_q_lora_rank: Optional[int] = None
    mla_qk_rope_head_dim: Optional[int] = None


def _build_attention(args: LMWidenArgs, head_dim, n_heads, n_kv_heads):
    if args.arch_type == "linear":
        return LinearAttention(args.dim, head_dim, n_heads, n_kv_heads, args.rope_theta)
    if args.arch_type == "tokenformer":
        return TokenformerAttention(
            args.dim, head_dim, n_heads, n_kv_heads, args.rope_theta,
            args.pattention_num_tokens,
        )
    if args.arch_type == "diff":
        return DiffAttention(
            args.dim, head_dim, n_heads, n_kv_heads, args.rope_theta,
            args.diff_lambda_init,
        )
    if args.arch_type == "gla":
        return GatedLinearAttention(
            args.dim, head_dim, n_heads, n_kv_heads, args.rope_theta, args.gla_feature,
        )
    if args.arch_type == "mla":
        rope_dim = args.mla_qk_rope_head_dim or head_dim
        kv_lora = args.mla_kv_lora_rank or max(4 * head_dim, args.dim // 4)
        q_lora = args.mla_q_lora_rank or max(kv_lora, args.dim // 2)
        return MLAttention(
            args.dim, head_dim, n_heads, args.rope_theta, kv_lora, q_lora, rope_dim,
        )
    if args.arch_type == "mamba2":
        return MambaMixer(args)
    if args.arch_type == "fastrnn":
        return MinGRUMixer(args)
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
        # PaLM/GPT-J parallel block: attention and FFN read the SAME residual input
        # (two norms, both used -> no unused-param FSDP grad issue) and are summed.
        self.parallel = args.arch_type == "parallel"

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        if self.parallel:
            attn = self.attention(
                self.attention_norm(x), freq_cis, tok_idx=tok_idx, mask=mask,
                attn_impl=attn_impl,
            )
            return x + attn + self.feed_forward(self.ffn_norm(x))
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
        for loop_i in range(self.n_loops):
            for layer in self.layers:
                # Generation KV-cache must be loop-aware for weight-tied recurrence:
                # the SAME attention module is invoked n_loops times per token, and
                # each loop needs its own K/V history (loop-i queries attend to loop-i
                # keys of earlier positions). The generator attaches a LoopedKVCache
                # with n_loops slots; tell it which loop we are in. A plain (single-
                # slot) KVCache would keep only the last loop's K/V -> garbage decode.
                # During training there is no kv_cache, so this is a no-op.
                attn = layer.attention
                if hasattr(attn, "kv_cache") and hasattr(attn.kv_cache, "_loop_idx"):
                    attn.kv_cache._loop_idx = loop_i
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
    # The mamba2 mixer registers a fused custom op (mamba_ssm::ssm_chunk_scan_combined_fwd).
    # When selective activation checkpointing is on (the FineWeb config), exclude it from
    # recompute exactly as apps/mamba does. We start from lingua's default set (so the
    # attention archs keep their flash/sdpa exclusions unchanged) and just add the mamba
    # op. Returns None when the kernel package is absent so the default policy applies.
    try:
        import apps.mamba.component.ssm_compilable  # noqa: F401 (registers the op)
        from lingua.distributed import default_no_recompute_ops

        return set(default_no_recompute_ops) | {
            torch.ops.mamba_ssm.ssm_chunk_scan_combined_fwd.default,
        }
    except Exception:
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
