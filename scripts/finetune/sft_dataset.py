from __future__ import annotations

import sys
from pathlib import Path


DLLM_ROOT = Path(__file__).resolve().parents[2] / "dllm"
if str(DLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(DLLM_ROOT))

from dllm.data import load_sft_dataset  # noqa: E402


ALPACA_PROMPT = (
    "Below is an instruction that describes a task, "
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def build_alpaca_prompt(instruction: str, input_text: str | None = None) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return ALPACA_PROMPT.format(instruction=instruction, input=input_text)
    return ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)


def load_dataset_for_ar(dataset_spec: str, load_preprocessed_data: bool = False):
    return load_sft_dataset(
        dataset_spec,
        load_preprocessed_data=load_preprocessed_data,
    )


def message_text(message: dict) -> str:
    parts = []
    reasoning = (message.get("reasoning_content") or "").strip()
    if reasoning:
        parts.append(reasoning)
    content = (message.get("content") or "").strip()
    if content:
        parts.append(content)
    return "\n\n".join(parts)


def render_messages_prompt(messages: list[dict]) -> str:
    sections = []
    for message in messages:
        text = message_text(message)
        if not text:
            continue
        role = str(message.get("role", "user")).strip().capitalize() or "User"
        sections.append(f"### {role}:\n{text}")
    sections.append("### Assistant:\n")
    return "\n\n".join(sections)


def split_messages(messages: list[dict]) -> tuple[list[dict], str]:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Expected at least one prompt message and one assistant reply.")

    final_message = messages[-1]
    if final_message.get("role") != "assistant":
        raise ValueError("Expected the final message to be an assistant response.")

    response = message_text(final_message)
    if not response:
        raise ValueError("Expected a non-empty assistant response.")
    return messages[:-1], response


def extract_prompt_response(example: dict) -> tuple[str, str]:
    if "messages" in example:
        messages = example["messages"]
        prompt_messages, response = split_messages(messages)

        if (
            len(prompt_messages) == 1
            and prompt_messages[0].get("role") == "user"
            and message_text(prompt_messages[0])
        ):
            prompt = build_alpaca_prompt(message_text(prompt_messages[0]))
        else:
            prompt = render_messages_prompt(prompt_messages)
        return prompt, response

    if "prompt" in example and "response" in example:
        prompt = build_alpaca_prompt(str(example.get("prompt", "")).strip())
        response = str(example.get("response", "")).strip()
        return prompt, response

    if any(key in example for key in ["instruction", "input", "output"]):
        prompt = build_alpaca_prompt(
            str(example.get("instruction", "")).strip(),
            str(example.get("input", "")).strip(),
        )
        response = str(example.get("output", "")).strip()
        return prompt, response

    if "problem" in example and (
        "generated_solution" in example or "solution" in example
    ):
        prompt = build_alpaca_prompt(str(example.get("problem", "")).strip())
        response = str(
            example.get("generated_solution", example.get("solution", ""))
        ).strip()
        return prompt, response

    if "question" in example and (
        "answer" in example or "response" in example or "output" in example
    ):
        prompt = build_alpaca_prompt(str(example.get("question", "")).strip())
        response = str(
            example.get(
                "answer",
                example.get("response", example.get("output", "")),
            )
        ).strip()
        return prompt, response

    raise ValueError(f"Unsupported SFT example schema with keys: {sorted(example.keys())}")


def tokenize_sft_example(example: dict, tokenizer, max_length: int) -> dict:
    if "input_ids" in example and "labels" in example:
        return example

    prompt, response = extract_prompt_response(example)
    full_text = prompt + response + tokenizer.eos_token

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )["input_ids"]

    labels = tokenized["input_ids"].copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized
