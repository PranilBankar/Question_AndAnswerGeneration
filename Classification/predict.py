"""
predict.py — Inference wrapper for the fine-tuned BERT chapter classifier

Usage:
    from Classification.predict import ClassifierPredictor
    predictor = ClassifierPredictor()
    result = predictor.predict("Which organelle is the powerhouse of the cell?")
    print(result)
    # → {"chapter": "Cell: The Unit of Life", "chapter_id": 8, "confidence": 0.97, "top3": [...]}
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# CONFIG
# ==============================
DEFAULT_MODEL_DIR  = "Classification/checkpoints/best_model"
LABEL_MAP_PATH     = "Classification/data/label_map.json"
MAX_LENGTH         = 128
CONFIDENCE_WARNING = 0.6   # Warn if top prediction confidence is below this


class ClassifierPredictor:
    """
    Loads a fine-tuned bert-base-uncased checkpoint and predicts
    the NCERT Biology chapter for a given question.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        print(f"[Classifier] Loading model from: {model_dir}")

        # Load label map {str(chapter_id): chapter_name}
        with open(LABEL_MAP_PATH) as f:
            raw_map = json.load(f)
        # Keys from JSON are strings — convert to int
        self.label_map = {int(k): v for k, v in raw_map.items()}

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Classifier] Using device: {self.device}")

        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(self.device)

        print(f"[Classifier] Ready — {len(self.label_map)} chapters.")

    def predict(self, question: str, top_k: int = 3) -> dict:
        """
        Predicts the chapter for a given NEET Biology question.

        Returns:
        {
            "chapter"    : "Biomolecules",
            "chapter_id" : 9,
            "confidence" : 0.94,
            "top3"       : [
                {"chapter": "Biomolecules",      "chapter_id": 9, "confidence": 0.94},
                {"chapter": "Cell Organelles",   "chapter_id": 8, "confidence": 0.04},
                {"chapter": "The Living World",  "chapter_id": 1, "confidence": 0.01},
            ]
        }
        """
        inputs = self.tokenizer(
            question,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits       # (1, num_labels)

        probs = F.softmax(logits, dim=-1).squeeze(0)  # (num_labels,)

        # Top-k predictions
        top_probs, top_ids = torch.topk(probs, k=min(top_k, len(self.label_map)))

        top_predictions = []
        for prob, label_id in zip(top_probs.tolist(), top_ids.tolist()):
            top_predictions.append({
                "chapter"    : self.label_map.get(label_id, f"Unknown({label_id})"),
                "chapter_id" : label_id,
                "confidence" : round(prob, 4),
            })

        best = top_predictions[0]

        if best["confidence"] < CONFIDENCE_WARNING:
            print(
                f"[Classifier] ⚠️  Low confidence ({best['confidence']:.2f}) for question:\n"
                f"    '{question}'\n"
                f"    Consider checking if this chapter is in the training data."
            )

        return {
            "chapter"    : best["chapter"],
            "chapter_id" : best["chapter_id"],
            "confidence" : best["confidence"],
            "top3"       : top_predictions,
        }


# ==============================
# QUICK TEST
# ==============================
if __name__ == "__main__":
    predictor = ClassifierPredictor()

    test_questions = [
        "What is the role of mitochondria in ATP synthesis?",
        "Explain the process of endospore formation in bacteria.",
        "What is the significance of the two-kingdom classification?",
    ]

    for q in test_questions:
        result = predictor.predict(q)
        print(f"\nQ: {q}")
        print(f"→ Chapter: {result['chapter']} (confidence: {result['confidence']:.2%})")
        top3_display = [(p['chapter'], f"{p['confidence']:.2%}") for p in result['top3']]
        print(f"  Top 3: {top3_display}")
