"""
dataset_multilabel.py — PyTorch Dataset for Multi-Label NEET Biology classification

Expected JSON format:
[
  {
    "question"    : "Which structures are common to both plant cells and animal cells?",
    "chapter_ids" : [7, 8]  # list of ints
  },
  ...
]
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NEETMultiLabelDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, num_labels: int, max_length: int = 128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        
        # This will handle both string dicts or lists of IDs depending on generation
        # We expect item["chapter_ids"] to be a list of integers
        label_ids = item.get("chapter_ids", [])
        if not isinstance(label_ids, list):
            label_ids = [label_ids]
            
        # Create a one-hot float vector for BCEWithLogitsLoss
        labels = torch.zeros(self.num_labels, dtype=torch.float)
        for c_id in label_ids:
            if 0 <= c_id < self.num_labels:
                labels[c_id] = 1.0

        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }
