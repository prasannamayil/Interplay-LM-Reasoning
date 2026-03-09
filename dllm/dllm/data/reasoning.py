from __future__ import annotations

from datasets import DatasetDict, load_dataset


def _resolve_reasoning_pair(example: dict) -> tuple[str, str]:
    if "problem" in example and (
        "generated_solution" in example or "solution" in example
    ):
        prompt = str(example.get("problem", "")).strip()
        response = str(
            example.get("generated_solution", example.get("solution", ""))
        ).strip()
        return prompt, response

    if "question" in example and (
        "answer" in example or "response" in example or "output" in example
    ):
        prompt = str(example.get("question", "")).strip()
        response = str(
            example.get(
                "answer",
                example.get("response", example.get("output", "")),
            )
        ).strip()
        return prompt, response

    if "instruction" in example and "output" in example:
        prompt = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "")).strip()
        if input_text:
            prompt = f"{prompt}\n\n{input_text}"
        response = str(example.get("output", "")).strip()
        return prompt, response

    raise ValueError(
        f"Unsupported reasoning SFT schema with keys: {sorted(example.keys())}"
    )


def load_dataset_reasoning_sft(dataset_name_or_path: str) -> DatasetDict:
    dataset = load_dataset(dataset_name_or_path)

    split_names = list(dataset.keys())
    cols_to_remove = dataset[split_names[0]].column_names

    def map_fn(example):
        prompt, response = _resolve_reasoning_pair(example)
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }

    dataset = dataset.map(map_fn, remove_columns=cols_to_remove, num_proc=4)

    if "train" in dataset and "test" in dataset:
        return DatasetDict({"train": dataset["train"], "test": dataset["test"]})
    if "train" in dataset and "validation" in dataset:
        return DatasetDict({"train": dataset["train"], "test": dataset["validation"]})
    if "train" in dataset:
        return DatasetDict(dataset["train"].train_test_split(test_size=0.05, seed=42))

    primary_split = split_names[0]
    return DatasetDict(dataset[primary_split].train_test_split(test_size=0.05, seed=42))
