"""
A2D-GPTNeoX: Adapt-to-Diffusion wrapper for GPT-NeoX / Pythia models.

Replaces causal attention with bidirectional (padding-only) attention,
enabling use as a backbone for MDLM / BD3LM diffusion language models.

Follows the same pattern as A2DLlamaModel / A2DQwen2Model.
"""

from typing import Optional

import torch
from torch import nn

import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor


class A2DGPTNeoXConfig(transformers.GPTNeoXConfig):
    model_type = "a2d-gpt_neox"


class A2DGPTNeoXModel(transformers.GPTNeoXModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # Diffusion models don't use KV caching
        use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        if cache_position is None:
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # --- Bidirectional (padding-only) mask instead of causal ---
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        if not (
            isinstance(attention_mask, BlockMask)
            or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
        ):
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype
            )

        hidden_states = self.emb_dropout(inputs_embeds)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            # GPT-NeoX layers return (hidden_states, present_key_value)
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


class A2DGPTNeoXLMHeadModel(transformers.GPTNeoXForCausalLM):
    config: A2DGPTNeoXConfig

    def __init__(self, config):
        transformers.GPTNeoXPreTrainedModel.__init__(self, config)
        self.gpt_neox = A2DGPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


transformers.AutoConfig.register("a2d-gpt_neox", A2DGPTNeoXConfig)
transformers.AutoModel.register(A2DGPTNeoXConfig, A2DGPTNeoXLMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DGPTNeoXConfig, A2DGPTNeoXLMHeadModel)


if __name__ == "__main__":
    import dllm
    from transformers import AutoModel

    config_path = dllm.utils.resolve_with_base_env(
        "EleutherAI/pythia-410m", "BASE_MODELS_DIR"
    )
    config = A2DGPTNeoXConfig.from_pretrained(config_path)
    for attr in ("auto_map", "architectures"):
        if hasattr(config, attr):
            delattr(config, attr)

    torch.set_default_device("cuda")
    model = A2DGPTNeoXLMHeadModel(config)
    model.save_pretrained("models-tmp/a2d-gpt_neox")
    auto_model = AutoModel.from_pretrained("models-tmp/a2d-gpt_neox")
