"""
train_multilabel.py — Fine-tune bert-base-uncased for Multi-Label NEET classification
"""

import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from Classification.dataset_multilabel import NEETMultiLabelDataset

# ==============================
# CONFIG
# ==============================
CONFIG_PATH = "Classification/config.json"
DATA_PATH   = "Classification/data/multilabel_generated_questions.json"
LABEL_MAP   = "Classification/data/label_map.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = json.load(f)

MODEL_NAME  = cfg["model_name"]
MAX_LENGTH  = cfg["max_length"]
BATCH_SIZE  = cfg["batch_size"]
LR          = cfg["learning_rate"]
EPOCHS      = cfg["num_epochs"]
OUTPUT_DIR  = "Classification/checkpoints_multilabel"
SEED        = cfg["seed"]


print("Loading label map...")
with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)
num_labels = len(label_map)
cfg["num_labels"] = num_labels

print(f"Loading data from {DATA_PATH}...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing {DATA_PATH}. Generate the data first.")

with open(DATA_PATH, encoding="utf-8") as f:
    all_data = json.load(f)

# Stratification is tricky for multi-label, simple split used here
train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=SEED)

os.makedirs("Classification/data", exist_ok=True)
train_path = "Classification/data/train_split_multilabel.json"
val_path   = "Classification/data/val_split_multilabel.json"

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)
with open(val_path, "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2)

print(f"Train: {len(train_data)} | Val: {len(val_data)}")

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = NEETMultiLabelDataset(train_path, tokenizer, num_labels, MAX_LENGTH)
val_dataset   = NEETMultiLabelDataset(val_path, tokenizer, num_labels, MAX_LENGTH)

print(f"Loading {MODEL_NAME} for multi-label classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# Custom metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Apply sigmoid
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    
    # Threshold at 0.5
    preds = np.zeros(probs.shape)
    preds[np.where(probs >= 0.5)] = 1
    
    # Macro F1 is standard for multi-label
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1_macro": macro_f1}

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
    metric_for_best_model="f1_macro",
    logging_steps=cfg["logging_steps"],
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nStarting multi-label training...")
trainer.train()

best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)
print(f"\n✅ Best multi-label model saved to: {best_model_dir}")

metrics = trainer.evaluate()
print(f"\nFinal Eval Metrics: {metrics}")
