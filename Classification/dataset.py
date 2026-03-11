"""
dataset.py — PyTorch Dataset for NEET Biology question classification

Expected JSON format for training data (see data/sample_questions.json):
[
  {
    "question"   : "Which organelle is the powerhouse of the cell?",
    "chapter"    : "Cell: The Unit of Life",
    "chapter_id" : 8     ← integer label, must start from 0, contiguous
  },
  ...
]

chapter_id mapping MUST be consistent and saved separately in label_map.json.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NEETDataset(Dataset):
    """
    Tokenizes NEET Biology questions for BertForSequenceClassification.
    """

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 128):
        """
        Args:
            data_path  : Path to JSON file with labeled questions.
            tokenizer  : HuggingFace tokenizer (bert-base-uncased).
            max_length : Max token length — 128 is fine for single questions.
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item     = self.data[idx]
        question = item["question"]
        label    = item["chapter_id"]   # integer class label

        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask" : encoding["attention_mask"].squeeze(0),  # (max_length,)
            "labels"         : torch.tensor(label, dtype=torch.long),
        }


def build_label_map(data_path: str, output_path: str = None) -> dict:
    """
    Scans the dataset and builds a {chapter_id: chapter_name} mapping.
    Saves it to output_path as JSON if provided.

    Run this once before training to verify your label space.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = {}
    for item in data:
        cid  = item["chapter_id"]
        name = item["chapter"]
        if cid not in label_map:
            label_map[cid] = name

    # Sort by label integer for readability
    label_map = dict(sorted(label_map.items()))

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)
        print(f"Label map saved to {output_path}")

    return label_map


if __name__ == "__main__":
    # Quick sanity check
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ds = NEETDataset("Classification/data/sample_questions.json", tokenizer)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"input_ids shape : {sample['input_ids'].shape}")
    print(f"label           : {sample['labels'].item()}")

    label_map = build_label_map(
        "Classification/data/sample_questions.json",
        output_path="Classification/data/label_map.json"
    )
    print(f"\nLabel map ({len(label_map)} classes):")
    for cid, name in label_map.items():
        print(f"  {cid}: {name}")
