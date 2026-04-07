def compute_confidence(
    retrieval_scores: list[float],
    verified: bool,
) -> float:
    """
    Computes a single confidence score [0.0 – 1.0] from two signals:

      - retrieval_quality (60%): average cosine similarity of retrieved chunks.
        High similarity means the question closely matched NCERT content.

      - verifier_bonus (40%): whether the LLM verifier confirmed the answer
        is supported by the retrieved context. Binary boost.

    Note: The BERT classifier confidence will be incorporated once the
    classification layer (Phase 3) is built. The formula will then become:
      0.4 * bert_conf + 0.35 * avg_retrieval + 0.25 * verifier

    Args:
        retrieval_scores: list of cosine similarity floats from pgvector (0–1)
        verified: bool from verifier.py (did the answer pass the faithfulness check)

    Returns:
        confidence: float in [0.0, 1.0]
    """
    if not retrieval_scores:
        avg_retrieval = 0.0
    else:
        # avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
        avg_retrieval = max(retrieval_scores)

    verifier_score = 1.0 if verified else 0.0

    confidence = (0.6 * avg_retrieval) + (0.4 * verifier_score)

    # Clamp to [0, 1] for safety
    return round(min(max(confidence, 0.0), 1.0), 4) 
