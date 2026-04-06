from sentence_transformers import CrossEncoder

_nli_model = None

def get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        print("[NLI] Loading NLI Verification model (cross-encoder/nli-distilroberta-base)...")
        _nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base', tokenizer_args={'use_fast': False})
        
    return _nli_model

import re

def verify_answer_nli(answer: str, chunks: list[dict]) -> dict:
    """
    Verifies if the generated answer is supported by the provided chunks using an NLI CrossEncoder.
    Now uses multi-hop reasoning (aggregates top chunks) and sentence-by-sentence evaluation.
    Returns: {"verified": bool, "explanation": str}
    """
    if not chunks:
        return {"verified": False, "explanation": "No context chunks provided to establish entailment."}
        
    model = get_nli_model()
    
    # --- Solution 3: Aggregate Context (Multi-Hop) ---
    # Combine the top 3 chunks into a single premise
    combined_premise = " ".join([c.get("text_content", "") for c in chunks[:3] if c.get("text_content")])
    if not combined_premise.strip():
        return {"verified": False, "explanation": "No valid text content found in retrieved chunks."}
        
    # --- Solution 2: Evaluate Sentence-by-Sentence ---
    # Split the generated answer into individual sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    if not sentences:
        sentences = [answer] # Fallback if splitting fails
        
    # Build batch pairs to evaluate all sentences at once, saving massive time
    pairs = [(combined_premise, sent) for sent in sentences]
    scores_list = model.predict(pairs)
        
    explanations = []
    has_contradiction = False
    
    # Check entailment for each sentence individually
    for sent, scores in zip(sentences, scores_list):
        contradiction = scores[0]
        entailment = scores[1]
        neutral = scores[2]
        
        # Log explanation for each sentence
        explanations.append(f"[{sent[:25]}...] E:{entailment:.2f} C:{contradiction:.2f}")
        
        # If any sentence is contradicted by the premise, fail the whole answer
        if contradiction > entailment:
            has_contradiction = True
    
    # Verified only if NO sentences contradict the text
    verified = not has_contradiction
    
    return {
        "verified": verified,
        "explanation": " | ".join(explanations)
    }
