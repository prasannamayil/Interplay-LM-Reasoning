"""
BD3LM eval harness with loglikelihood and generate_until.

Supports two likelihood estimators (controlled by ``ll_method`` kwarg):

- **elbo** (default): Monte Carlo ELBO — random masking + importance weighting,
  averaged over ``mc_num`` samples.
- **duel**: Exact likelihood via deterministic unmasking (Turok et al., 2026).
  Within each block of size ``block_size``, iteratively unmask positions
  according to a deterministic rule; blocks are processed autoregressively
  (preceding blocks provide context).  No MC averaging needed.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.a2d.eval).
"""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from tqdm import tqdm

from dllm.core.eval.base import BaseEvalConfig, BaseEvalHarness
from dllm.core.eval.mdlm import DUEL_RULE_FNS, DUEL_RULES
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
from dllm.core.trainers.bd3lm import _create_bd3lm_attention_mask


@dataclass
class BD3LMEvalSamplerConfig(BD3LMSamplerConfig):
    """Default sampler config for BD3LM eval."""

    max_new_tokens: int = 128
    steps: int = 128
    block_size: int = 32


@dataclass
class BD3LMEvalConfig(BaseEvalConfig):
    """Eval-only config for BD3LM."""

    max_length: int = 2048
    batch_size: int = 32
    mc_num: int = 128
    block_size: int = 32
    ll_method: str = "elbo"
    duel_rule: str = "prob_margin"
    duel_k: int = 1


class BD3LMEvalHarness(BaseEvalHarness):
    """
    BD3LM eval: loglikelihood via MC ELBO or DUEL + generate_until via BD3LMSampler.
    """

    def __init__(
        self,
        eval_config: BD3LMEvalConfig | None = None,
        sampler_config: BD3LMSamplerConfig | None = None,
        sampler_cls: type[BD3LMSampler] = BD3LMSampler,
        **kwargs,
    ):
        eval_config = eval_config or BD3LMEvalConfig()
        sampler_config = sampler_config or BD3LMEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )

        self.mask_id = self.tokenizer.mask_token_id
        self.max_length = int(kwargs.get("max_length", eval_config.max_length))
        self.mc_num = int(kwargs.get("mc_num", eval_config.mc_num))
        self.block_size_ll = int(kwargs.get("block_size", eval_config.block_size))

        self.ll_method: Literal["elbo", "duel"] = str(
            kwargs.get("ll_method", eval_config.ll_method)
        ).lower()
        self.duel_rule = str(
            kwargs.get("duel_rule", eval_config.duel_rule)
        ).lower()
        self.duel_k = int(kwargs.get("duel_k", eval_config.duel_k))

        assert self.ll_method in ("elbo", "duel"), (
            f"ll_method must be 'elbo' or 'duel', got '{self.ll_method}'"
        )
        assert self.duel_rule in DUEL_RULES, (
            f"duel_rule must be one of {DUEL_RULES}, got '{self.duel_rule}'"
        )
        if self.ll_method == "elbo":
            assert self.mc_num % self.batch_size == 0

    # ── Private helpers ──────────────────────────────────────────────

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _pad_to_block(self, seq: torch.Tensor) -> torch.Tensor:
        """Pad sequence length to a multiple of block_size with EOS tokens."""
        b, l = seq.shape
        bs = self.block_size_ll
        target = ((l + bs - 1) // bs) * bs
        if target > l:
            pad = torch.full(
                (b, target - l),
                self.tokenizer.eos_token_id,
                dtype=seq.dtype,
                device=seq.device,
            )
            seq = torch.cat([seq, pad], dim=1)
        return seq

    def _forward_process(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask random subset of target tokens (same as MDLM forward process)."""
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b, int(prompt_index.sum()), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        p_mask = (x / target_len).unsqueeze(1).repeat(1, l)
        return noisy_batch, p_mask

    @torch.no_grad()
    def _get_logits_bd3lm(
        self, noisy_seq: torch.Tensor, clean_seq: torch.Tensor
    ) -> torch.Tensor:
        """BD3LM forward: concat [x_t || x_0], apply block-diagonal mask, return logits for x_t half."""
        b, l = noisy_seq.shape
        concat_input_ids = torch.cat([noisy_seq, clean_seq], dim=1)

        attn_impl = getattr(self.model.config, "_attn_implementation", "sdpa")
        if attn_impl == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            attention_mask = create_block_mask(
                partial(
                    _create_bd3lm_attention_mask, block_size=self.block_size_ll, n=l
                ),
                B=None,
                H=None,
                Q_LEN=l * 2,
                KV_LEN=l * 2,
            )
        else:
            q_idx = torch.arange(l * 2, device=noisy_seq.device)[:, None]
            kv_idx = torch.arange(l * 2, device=noisy_seq.device)[None, :]
            attention_mask = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=q_idx,
                kv_idx=kv_idx,
                block_size=self.block_size_ll,
                n=l,
            )
            attention_mask = (
                attention_mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(1, 1, 2 * l, 2 * l)
                .to(noisy_seq.device)
            )

        base_pos = torch.arange(l, device=noisy_seq.device).unsqueeze(0).expand(b, l)
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)

        outputs = self.model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        logits = outputs.logits[:, :l]
        return logits

    @torch.no_grad()
    def _get_loglikelihood_elbo(
        self, prefix: torch.Tensor, target: torch.Tensor
    ) -> float:
        """Monte Carlo ELBO estimate using BD3LM's block-diagonal forward pass."""
        seq = torch.cat([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        seq = self._pad_to_block(seq)

        b, l = seq.shape
        prompt_index = torch.arange(l, device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self._get_logits_bd3lm(perturbed_seq, seq)
            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _get_loglikelihood_duel(
        self, prefix: torch.Tensor, target: torch.Tensor
    ) -> float:
        """DUEL exact likelihood for BD3LM (Turok et al., 2026).

        Processes blocks autoregressively: preceding blocks are fully revealed
        (clean x_0 context).  Within each block, deterministic unmasking
        accumulates log p(true token) at each step.
        """
        clean = torch.cat([prefix, target]).to(self.device)
        seq_1d = clean.clone()
        seq_1d = self._pad_to_block(seq_1d.unsqueeze(0)).squeeze(0)
        padded_len = seq_1d.shape[0]
        bs = self.block_size_ll

        noisy = seq_1d.clone()
        target_start = len(prefix)
        noisy[target_start:] = self.mask_id

        rule_fn = DUEL_RULE_FNS[self.duel_rule]
        ll = 0.0

        num_blocks = padded_len // bs
        for block_idx in range(num_blocks):
            block_start = block_idx * bs
            block_end = block_start + bs

            block_masked = (noisy[block_start:block_end] == self.mask_id)
            if not block_masked.any():
                continue

            while True:
                masked_local = (noisy[block_start:block_end] == self.mask_id).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if masked_local.numel() == 0:
                    break

                noisy_batch = noisy.unsqueeze(0)
                clean_batch = seq_1d.unsqueeze(0)
                logits = self._get_logits_bd3lm(noisy_batch, clean_batch)
                probs = torch.softmax(logits[0].float(), dim=-1)

                global_masked = masked_local + block_start
                k = min(self.duel_k, masked_local.numel())
                selected_local = rule_fn(masked_local, probs[block_start:block_end], k)
                selected_global = selected_local + block_start

                for pos in selected_global:
                    ll += torch.log(probs[pos, seq_1d[pos]] + 1e-30).item()

                noisy[selected_global] = seq_1d[selected_global]

        return ll

    @torch.no_grad()
    def _get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        if self.ll_method == "duel":
            return self._get_loglikelihood_duel(prefix, target)
        return self._get_loglikelihood_elbo(prefix, target)

    # ── Public API (lm-eval interface) ────────────────────────────────

    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        method_label = (
            f"DUEL ({self.duel_rule}, k={self.duel_k})"
            if self.ll_method == "duel"
            else f"MC ELBO (mc_num={self.mc_num})"
        )
        out = []
        for instance in tqdm(requests, desc=f"BD3LM loglikelihood [{method_label}]..."):
            context_enc, continuation_enc = self._encode_pair(*instance.args)
            assert len(context_enc) + len(continuation_enc) <= self.max_length, (
                f"Context + continuation length exceeds {self.max_length} tokens: "
                f"{len(context_enc)} + {len(continuation_enc)}"
            )

            context = torch.tensor(context_enc, device=self.device, dtype=torch.long)
            continuation = torch.tensor(
                continuation_enc, device=self.device, dtype=torch.long
            )

            logprob = self._get_loglikelihood(context, continuation)
            out.append((logprob, False))
        torch.cuda.empty_cache()
        return out
