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
    
    # Loop through top 5 chunks to see if ANY chunk successfully entails the answer.
    # This prevents the 512 token limit from hiding evidence located in chunk 3 or 4.
    best_entailment = -999.0
    best_scores = [0, 0, 0]
    
    for c in chunks[:5]:
        premise = c.get("text_content", "")
        if not premise: continue
        
        # Predict pair
        scores = model.predict([(premise, answer)])[0]
        entailment_score = scores[1]
        
        if entailment_score > best_entailment:
            best_entailment = entailment_score
            best_scores = scores
            
    contradiction = best_scores[0]
    entailment    = best_scores[1]
    neutral       = best_scores[2]
    
    # Simple deterministic gate: Entailment vs Contradiction
    verified = bool(entailment > contradiction)
    
    explanation = f"NLI Max Scores -> Entailment: {entailment:.2f} | Contradiction: {contradiction:.2f} | Neutral: {neutral:.2f}"
    
    return {
        "verified": verified,
        "explanation": explanation
    }
