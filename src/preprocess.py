"""GSM8K preprocessing pipeline – strict label-leak prevention."""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def extract_numeric_answer(ans: str) -> str:
    """Return substring after first '####' or whole string if absent."""
    return ans.split("####")[-1].strip()


# -----------------------------------------------------------------------------
# Tokeniser
# -----------------------------------------------------------------------------

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=str(CACHE_DIR), use_fast=True)
    # ensure PAD token exists and left-pad (for generation compatibility)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


# -----------------------------------------------------------------------------
# Example builders
# -----------------------------------------------------------------------------

def _build_train_example(ex: Dict[str, str], tokenizer, max_len: int) -> Dict[str, Any]:
    prompt = ex["question"].strip() + "\n\nAnswer:\n####"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_text = extract_numeric_answer(ex["answer"]) + tokenizer.eos_token
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    placeholder_ids = [tokenizer.pad_token_id] * len(answer_ids)  # prevents label leakage

    input_ids = (prompt_ids + placeholder_ids)[-max_len:]
    attention_mask = [1] * len(input_ids)

    labels = ([-100] * len(prompt_ids) + answer_ids)[-max_len:]
    if len(labels) < len(input_ids):
        labels = [-100] * (len(input_ids) - len(labels)) + labels

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _build_eval_example(ex: Dict[str, str], tokenizer, max_len: int) -> Dict[str, Any]:
    prompt = ex["question"].strip() + "\n\nAnswer:\n####"
    enc = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_len)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": extract_numeric_answer(ex["answer"]),
    }


# -----------------------------------------------------------------------------
# Collator
# -----------------------------------------------------------------------------

def _collate_fn(tokenizer, is_train: bool):
    def collate(features: List[Dict[str, Any]]):
        input_ids = [f["input_ids"] for f in features]
        attention = [f["attention_mask"] for f in features]
        batch_enc = tokenizer.pad({"input_ids": input_ids, "attention_mask": attention}, return_tensors="pt")

        if is_train:
            labels_list = [f["labels"] for f in features]
            max_len = max(len(l) for l in labels_list)
            labels_tensor = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
            for i, seq in enumerate(labels_list):
                labels_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            batch_enc["labels"] = labels_tensor
        else:
            batch_enc["labels"] = [f["labels"] for f in features]
        return batch_enc

    return collate


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_dataloaders(cfg, tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds = load_dataset(cfg.dataset.name, cfg.dataset.get("config", None), cache_dir=str(CACHE_DIR))

    split = ds["train"].train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    test_ds = ds["test"]

    max_len = cfg.dataset.max_seq_length

    train_ds = train_ds.map(lambda x: _build_train_example(x, tokenizer, max_len), remove_columns=train_ds.column_names)
    val_ds = val_ds.map(lambda x: _build_eval_example(x, tokenizer, max_len), remove_columns=val_ds.column_names)
    test_ds = test_ds.map(lambda x: _build_eval_example(x, tokenizer, max_len), remove_columns=test_ds.column_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=_collate_fn(tokenizer, is_train=True),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=_collate_fn(tokenizer, is_train=False),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=_collate_fn(tokenizer, is_train=False),
    )

    return train_loader, val_loader, test_loader
