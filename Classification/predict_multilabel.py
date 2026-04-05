"""
predict_multilabel.py — Inference wrapper for fine-tuned Multi-Label BERT
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_DIR  = "Classification/checkpoints_multilabel/best_model"
LABEL_MAP_PATH     = "Classification/data/label_map.json"
MAX_LENGTH         = 128
THRESHOLD          = 0.5 

class MultiLabelPredictor:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        print(f"[MultiLabel Predictor] Loading model from: {model_dir}")

        with open(LABEL_MAP_PATH) as f:
            raw_map = json.load(f)
        self.label_map = {int(k): v for k, v in raw_map.items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MultiLabel Predictor] Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, question: str) -> list:
        inputs = self.tokenizer(
            question,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Apply sigmoid for independent probabilities
        probs = torch.sigmoid(logits).squeeze(0)

        results = []
        for label_id, prob in enumerate(probs.tolist()):
            if prob >= THRESHOLD:
                results.append({
                    "chapter": self.label_map.get(label_id, f"Unknown({label_id})"),
                    "chapter_id": label_id,
                    "confidence": round(prob, 4),
                })
        
        # Sort by confidence
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        return results

if __name__ == "__main__":
    import sys
    try:
        predictor = MultiLabelPredictor()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    qs = ["Which structures are common to both plant cells and animal cells?"]
    for q in qs:
        print(f"\nQ: {q}")
        preds = predictor.predict(q)
        print("Predictions:")
        for p in preds:
            print(f" - {p['chapter']} : {p['confidence']:.2%}")
