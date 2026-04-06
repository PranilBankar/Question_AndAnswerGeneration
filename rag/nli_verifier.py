from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

_nli_model = None
_nli_tokenizer = None

# DeBERTa-v3-small: far better at scientific/technical text and logical reasoning
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
MAX_PREMISE_TOKENS = 400  # Reserve ~112 tokens for the hypothesis (answer sentence)

def get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        print(f"[NLI] Loading NLI Verification model ({NLI_MODEL_NAME})...")
        _nli_model = CrossEncoder(NLI_MODEL_NAME, tokenizer_args={'use_fast': False})
    return _nli_model

def get_nli_tokenizer():
    global _nli_tokenizer
    if _nli_tokenizer is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, use_fast=False)
    return _nli_tokenizer

import re

def _truncate_premise(text: str, max_tokens: int = MAX_PREMISE_TOKENS) -> str:
    """Truncate premise text to fit within the model's token budget."""
    tokenizer = get_nli_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def verify_answer_nli(answer: str, chunks: list[dict]) -> dict:
    """
    Verifies if the generated answer is supported by the provided chunks using an NLI CrossEncoder.
    Uses multi-hop reasoning (aggregates top chunks) and sentence-by-sentence evaluation.
    Returns: {"verified": bool, "explanation": str}
    """
    if not chunks:
        return {"verified": False, "explanation": "No context chunks provided to establish entailment."}
        
    model = get_nli_model()
    
    # --- Aggregate Context (Multi-Hop) ---
    # Combine the top 3 chunks into a single premise, then truncate to fit token budget
    combined_premise = " ".join([c.get("text_content", "") for c in chunks[:3] if c.get("text_content")])
    if not combined_premise.strip():
        return {"verified": False, "explanation": "No valid text content found in retrieved chunks."}
    combined_premise = _truncate_premise(combined_premise)
        
    # --- Evaluate Sentence-by-Sentence ---
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    if not sentences:
        sentences = [answer]
        
    # Batch all sentence pairs for a single model.predict() call
    pairs = [(combined_premise, sent) for sent in sentences]
    scores_list = model.predict(pairs)
    
    # DeBERTa-v3 NLI outputs: [entailment, neutral, contradiction] (different order from distilroberta!)
    explanations = []
    supported_sentences = 0
    
    for sent, scores in zip(sentences, scores_list):
        entailment    = scores[0]  # DeBERTa: index 0 = entailment
        neutral       = scores[1]  # DeBERTa: index 1 = neutral
        contradiction = scores[2]  # DeBERTa: index 2 = contradiction
        
        explanations.append(f"[{sent[:30]}...] E:{entailment:.2f} C:{contradiction:.2f} N:{neutral:.2f}")
        
        # Sentence is grounded if entailment beats contradiction
        if entailment > contradiction:
            supported_sentences += 1
            
    # Verified if the majority of sentences are grounded in the text
    verified = (supported_sentences / len(sentences)) >= 0.5
    
    return {
        "verified": verified,
        "explanation": " | ".join(explanations)
    }
