"""
train.py — Fine-tune bert-base-uncased for NEET Biology chapter classification

Flow:
  1. Load config from config.json
  2. Load and split the labeled dataset (80/20 train/val split)
  3. Load bert-base-uncased with a classification head
  4. Fine-tune using HuggingFace Trainer API
  5. Evaluate & save the best checkpoint

Run:
  python -m Classification.train

When you have your real labeled data, replace the path in DATA_PATH.
"""

import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from Classification.dataset import NEETDataset, build_label_map

# ==============================
# CONFIG
# ==============================
CONFIG_PATH = "Classification/config.json"
DATA_PATH   = "Classification/data/generated_questions.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = json.load(f)

MODEL_NAME  = cfg["model_name"]           # "bert-base-uncased"
MAX_LENGTH  = cfg["max_length"]           # 128
BATCH_SIZE  = cfg["batch_size"]           # 16
LR          = cfg["learning_rate"]        # 2e-5
EPOCHS      = cfg["num_epochs"]           # 5
OUTPUT_DIR  = cfg["output_dir"]           # "Classification/checkpoints"
SEED        = cfg["seed"]                 # 42


# ==============================
# STEP 1: Build label map
# ==============================
print("Loading label map...")
with open("Classification/data/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
num_labels = len(label_map)
print(f"Number of classes (chapters): {num_labels}")

# Update config num_labels dynamically
cfg["num_labels"] = num_labels


# ==============================
# STEP 2: Load & split data
# ==============================
with open(DATA_PATH, encoding="utf-8") as f:
    all_data = json.load(f)

train_data, val_data = train_test_split(
    all_data, test_size=0.2, random_state=SEED, stratify=[d["chapter_id"] for d in all_data]
)

# Save splits for transparency
os.makedirs("Classification/data", exist_ok=True)
with open("Classification/data/train_split.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)
with open("Classification/data/val_split.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2)

print(f"Train: {len(train_data)} | Val: {len(val_data)}")


# ==============================
# STEP 3: Tokenizer & Datasets
# ==============================
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Write splits to temp files (NEETDataset reads from JSON path)
train_dataset = NEETDataset("Classification/data/train_split.json", tokenizer, MAX_LENGTH)
val_dataset   = NEETDataset("Classification/data/val_split.json",   tokenizer, MAX_LENGTH)


# ==============================
# STEP 4: Load BERT Model
# ==============================
print(f"Loading {MODEL_NAME} with {num_labels} output labels...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
)


# ==============================
# STEP 5: Metrics
# ==============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1": f1}


# ==============================
# STEP 6: Training Arguments
# ==============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=cfg["warmup_ratio"],
    weight_decay=cfg["weight_decay"],
    evaluation_strategy=cfg["evaluation_strategy"],
    save_strategy=cfg["save_strategy"],
    load_best_model_at_end=cfg["load_best_model_at_end"],
    metric_for_best_model=cfg["metric_for_best_model"],
    logging_steps=cfg["logging_steps"],
    seed=SEED,
    report_to="none",     # set to "wandb" or "tensorboard" if you want tracking
)


# ==============================
# STEP 7: Train
# ==============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nStarting training...")
trainer.train()


# ==============================
# STEP 8: Save best model
# ==============================
best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)
print(f"\n✅ Best model saved to: {best_model_dir}")

# Save final eval metrics to Evaluation_metrics folder
metrics = trainer.evaluate()
print(f"\nFinal Eval Metrics: {metrics}")

eval_dir = "Classification/Evaluation_metrics"
os.makedirs(eval_dir, exist_ok=True)
with open(os.path.join(eval_dir, "eval_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"📊 Metrics saved to {eval_dir}/eval_metrics.json")

# Also save training history
log_history = trainer.state.log_history
with open(os.path.join(eval_dir, "training_log.json"), "w") as f:
    json.dump(log_history, f, indent=2)
print(f"📊 Training log saved to {eval_dir}/training_log.json")
