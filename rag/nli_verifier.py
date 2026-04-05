from sentence_transformers import CrossEncoder

_nli_model = None

def get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        print("[NLI] Loading NLI Verification model (cross-encoder/nli-distilroberta-base)...")
        _nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base', tokenizer_args={'use_fast': False})
    return _nli_model

def verify_answer_nli(answer: str, chunks: list[dict]) -> dict:
    """
    Verifies if the generated answer is supported by the provided chunks using an NLI CrossEncoder.
    Returns: {"verified": bool, "explanation": str}
    """
    if not chunks:
        return {"verified": False, "explanation": "No context chunks provided to establish entailment."}
        
    model = get_nli_model()
    
    # Combine the top 2 chunks as the "Premise" (limits token length to ~512)
    top_chunks = chunks[:2]
    premise = " ".join([c.get("text_content", "") for c in top_chunks])
    
    # Predict takes a list of pairs: [(Premise, Hypothesis)]
    scores = model.predict([(premise, answer)])[0]
    
    # For 'cross-encoder/nli-distilroberta-base':
    # scores[0] = Contradiction
    # scores[1] = Entailment
    # scores[2] = Neutral
    
    contradiction = scores[0]
    entailment    = scores[1]
    neutral       = scores[2]
    
    # A simple deterministic gate:
    # If entailment is stronger than contradiction, it's considered verified.
    # Optionally, we could require entailment > neutral, but often 'neutral' is high if the answer is short.
    verified = bool(entailment > contradiction)
    
    explanation = f"NLI Scores -> Entailment: {entailment:.2f} | Contradiction: {contradiction:.2f} | Neutral: {neutral:.2f}"
    
    return {
        "verified": verified,
        "explanation": explanation
    }
