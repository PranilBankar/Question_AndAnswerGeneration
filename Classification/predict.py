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
        
        # Calculate Normalized Entropy
        num_labels = len(self.label_map)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        normalized_entropy = entropy / torch.log(torch.tensor(num_labels, dtype=torch.float32)).item()

        # Top-k predictions
        top_probs, top_ids = torch.topk(probs, k=min(top_k, num_labels))

        top_predictions = []
        for prob, label_id in zip(top_probs.tolist(), top_ids.tolist()):
            top_predictions.append({
                "chapter"    : self.label_map.get(label_id, f"Unknown({label_id})"),
                "chapter_id" : label_id,
                "confidence" : round(prob, 4),
            })

        best = top_predictions[0]
        margin = top_probs[0].item() - top_probs[1].item() if len(top_probs) > 1 else 1.0

        # Heuristic Keyword Check (Expanded list)
        question_lower = question.lower()
        bio_keywords = [
            "cell", "dna", "plant", "animal", "human", "body", "blood", "heart", 
            "gene", "protein", "virus", "bacteria", "disease", "reproduction", 
            "organ", "tissue", "biology", "enzyme", "acid", "carbon", "oxygen", 
            "water", "bone", "muscle", "lung", "brain", "leaf", "root", "stem", 
            "flower", "seed", "fruit", "chromosome", "rna", "atp", "metabolism",
            "photosynthesis", "respiration", "digestion", "nervous", "hormone"
        ]
        has_bio_signal = any(kw in question_lower for kw in bio_keywords)

        # Dynamic Confidence Threshold
        # If the question contains no known biology words, be highly skeptical.
        min_confidence = 0.55 if has_bio_signal else 0.85

        # OOD Rejection Rules
        rejected = False
        rejection_reason = ""
        
        if best["confidence"] < min_confidence:
            rejected = True
            if not has_bio_signal:
                rejection_reason = f"No biological keywords found & Confidence too low ({best['confidence']:.2f} < 0.85 for generic text)"
            else:
                rejection_reason = f"Top-1 confidence too low ({best['confidence']:.2f} < 0.55)"
        elif margin < 0.12:
            rejected = True
            rejection_reason = f"Margin between top 2 chapters too small ({margin:.2f} < 0.12)"
        elif normalized_entropy > 0.86:
            rejected = True
            rejection_reason = f"Model uncertainty too high (Entropy {normalized_entropy:.2f} > 0.86)"

        if best["confidence"] < CONFIDENCE_WARNING and not rejected:
            print(
                f"[Classifier] ⚠️  Low confidence ({best['confidence']:.2f}) for question:\n"
                f"    '{question}'\n"
                f"    Consider checking if this chapter is in the training data."
            )

        return {
            "chapter"       : best["chapter"],
            "chapter_id"    : best["chapter_id"],
            "confidence"    : best["confidence"],
            "top3"          : top_predictions,
            "rejected"      : rejected,
            "rejection_reason": rejection_reason,
            "entropy"       : round(normalized_entropy, 4),
            "margin"        : round(margin, 4)
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
