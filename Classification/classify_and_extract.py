"""
classify_and_extract.py — BERT Chapter Classifier + Topic Extractor

Orchestrates:
  1. BERT classifier  → predicts which NCERT chapter a question belongs to
  2. Supabase retriever → finds relevant chunks within that chapter
  3. Topic extractor  → extracts unique topics/subtopics from retrieved chunks

Usage:
    from Classification.classify_and_extract import classify_question
    result = classify_question("What is the role of cofactors in enzyme activity?")
    print(result)

Returns:
    {
        "chapter": "Biomolecules",
        "chapter_id": 8,
        "chapter_confidence": 0.94,
        "topics": [
            {"section_code": "9.8", "section_title": "Enzymes", "similarity": 0.82},
            {"section_code": "9.8.6", "section_title": "Co-factors", "similarity": 0.71},
        ]
    }
"""

from Classification.predict import ClassifierPredictor
from rag.retriever import retrieve_chunks

# ==============================
# CONFIG
# ==============================
TOPIC_SIMILARITY_THRESHOLD = 0.30   # Minimum similarity to consider a topic relevant
MAX_TOPICS = 3                      # Maximum topics/subtopics to return
MIN_TOPICS = 1                      # Minimum topics to always return (top-1 fallback)
RETRIEVAL_TOP_K = 10                # Fetch more chunks to find diverse topics

# ==============================
# SINGLETON PREDICTOR
# ==============================
_predictor: ClassifierPredictor = None


def get_predictor() -> ClassifierPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ClassifierPredictor()
    return _predictor


# ==============================
# MAIN FUNCTION
# ==============================
def classify_question(
    question: str,
    similarity_threshold: float = TOPIC_SIMILARITY_THRESHOLD,
    max_topics: int = MAX_TOPICS,
    min_topics: int = MIN_TOPICS,
) -> dict:
    """
    Classify a NEET Biology question into chapter + topic/subtopic.

    Steps:
      1. BERT predicts the chapter (91.7% F1 accuracy)
      2. Retriever fetches top-k chunks filtered to that chapter
      3. Unique (section_code, section_title) pairs are extracted
      4. Filtered by similarity threshold, capped at max_topics
      5. At least min_topics are always returned (fallback to top-1)

    Args:
        question             : The student's question.
        similarity_threshold : Min similarity score for a topic to qualify.
        max_topics           : Maximum number of topics to return.
        min_topics           : Minimum number of topics to always return.

    Returns:
        {
            "chapter": str,
            "chapter_id": int,
            "chapter_confidence": float,
            "topics": [
                {"section_code": str, "section_title": str, "similarity": float},
                ...
            ]
        }
    """
    # ── Step 1: BERT Chapter Prediction ──
    predictor = get_predictor()
    bert_result = predictor.predict(question)

    chapter_name = bert_result["chapter"]
    chapter_id = bert_result["chapter_id"]
    chapter_confidence = bert_result["confidence"]

    print(f"[Classify] BERT → {chapter_name} (confidence: {chapter_confidence:.2%})")

    # ── Step 2: Retrieve chunks filtered by predicted chapter ──
    chunks = retrieve_chunks(
        question,
        chapter_filter=chapter_name,
        top_k=RETRIEVAL_TOP_K,
    )

    if not chunks:
        print("[Classify] ⚠️  No chunks retrieved — returning chapter only.")
        return {
            "chapter": chapter_name,
            "chapter_id": chapter_id,
            "chapter_confidence": chapter_confidence,
            "topics": [],
        }

    # ── Step 3: Extract unique topics, keeping highest similarity per topic ──
    topic_map = {}  # key: (section_code, section_title) → best similarity

    for chunk in chunks:
        section_code = chunk.get("section") or ""
        section_title = chunk.get("section_title") or ""
        similarity = chunk.get("similarity", 0.0)

        if not section_code and not section_title:
            continue

        key = (section_code, section_title)

        # Keep the highest similarity for each unique topic
        if key not in topic_map or similarity > topic_map[key]:
            topic_map[key] = similarity

    # Sort by similarity descending
    sorted_topics = sorted(topic_map.items(), key=lambda x: x[1], reverse=True)

    # ── Step 4: Apply threshold + min/max constraints ──
    filtered = [
        {
            "section_code": code,
            "section_title": title,
            "similarity": round(sim, 4),
        }
        for (code, title), sim in sorted_topics
        if sim >= similarity_threshold
    ]

    # Cap at max_topics
    filtered = filtered[:max_topics]

    # Fallback: if nothing passes threshold, always return top-1
    if len(filtered) < min_topics and sorted_topics:
        for i in range(min(min_topics, len(sorted_topics))):
            (code, title), sim = sorted_topics[i]
            entry = {
                "section_code": code,
                "section_title": title,
                "similarity": round(sim, 4),
            }
            if entry not in filtered:
                filtered.append(entry)
        filtered = filtered[:max_topics]

    print(f"[Classify] Topics found: {len(filtered)}")
    for t in filtered:
        print(f"  → {t['section_code']} - {t['section_title']} (sim: {t['similarity']:.4f})")

    return {
        "chapter": chapter_name,
        "chapter_id": chapter_id,
        "chapter_confidence": chapter_confidence,
        "topics": filtered,
    }


# ==============================
# QUICK TEST
# ==============================
if __name__ == "__main__":
    test_questions = [
        "What is the role of cofactors in enzyme activity?",
        "Explain the process of meiosis and its significance.",
        "What are the different types of immunity?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        result = classify_question(q)
        print(f"\n📗 Chapter: {result['chapter']} (confidence: {result['chapter_confidence']:.2%})")
        print(f"📝 Topics:")
        for t in result['topics']:
            print(f"   {t['section_code']} — {t['section_title']} (similarity: {t['similarity']:.4f})")
