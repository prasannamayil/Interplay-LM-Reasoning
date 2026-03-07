"""
Token-level NLL/PPL metrics for evaluation.

- NLLMetric: token-level mean negative log-likelihood (weighted mean over tokens).
- PPLMetric: exp(mean NLL) = perplexity.

Both use sync_on_compute=True so that compute() aggregates over all ranks.

Note for diffusion (MDLM/BD3LM): The trainer passes value=token_nll (nonzero only at
*masked* positions) and weight=maskable_mask (all valid positions). So the reported
"nll"/"ppl" are (sum of NLL over masked tokens) / (total valid tokens)—a training
proxy, NOT the sequence NLL/perplexity under an AR model. True sequence log-probability
under the diffusion model would require e.g. ELBO or sampling.
"""

import torch
import torchmetrics


class NLLMetric(torchmetrics.aggregation.MeanMetric):
    """Token-level mean NLL. Weights define the denominator (e.g. maskable or masked tokens)."""

    def __init__(self, **kwargs):
        # Ensure cross-rank aggregation when compute() is called
        kwargs.setdefault("sync_on_compute", True)
        super().__init__(**kwargs)


class PPLMetric(NLLMetric):
    """Token-level perplexity = exp(mean NLL)."""

    def compute(self) -> torch.Tensor:
        mean_nll = super().compute()
        return torch.exp(mean_nll)
