"""
pipeline.py  —  Main RAG orchestrator

Ties together:
  1. Retriever     → get top-k NCERT chunks from Supabase
  2. PromptBuilder → construct grounded LLM prompts
  3. Generator     → call Groq to answer / justify / verify
  4. Confidence    → produce a single confidence score

Usage:
    from rag.pipeline import run_pipeline
    result = run_pipeline("Which organelle is the powerhouse of the cell?")
    print(result)
"""

from rag.retriever      import retrieve_chunks
from rag.prompt_builder import (
    build_answer_prompt,
    build_justification_prompt,
)
from rag.generator  import generate_answer, generate_justification
from rag.nli_verifier import verify_answer_nli
from rag.confidence import compute_confidence
from Classification.classify_and_extract import classify_question


def run_pipeline(
    question: str,
    chapter_filter: str = None,
    top_k: int = 5,
    use_classifier: bool = False,
) -> dict:
    """
    End-to-end RAG pipeline for a NEET Biology question.

    Args:
        question       : The student's question string.
        chapter_filter : Optional chapter name to restrict retrieval scope.
                         Auto-populated by BERT classifier if use_classifier=True.
        top_k          : Number of NCERT chunks to retrieve.
        use_classifier : If True, auto-classify question to get chapter + topics.

    Returns a structured dict:
    {
        "question"       : str,
        "classification" : {chapter, chapter_id, chapter_confidence, topics},
        "answer"         : str,
        "justification"  : list[str],
        "sources"        : list[{chapter, section, section_title, text_content, similarity}],
        "confidence"     : float,
        "verified"       : bool,
        "verifier_note"  : str,
    }
    """

    # ── Step 0: BERT Classification + Topic Extraction ──────────────────────
    classification = None
    if use_classifier:
        print("[Pipeline] Step 0: BERT classification + topic extraction...")
        classification = classify_question(question)
        predicted_chapter = classification["chapter"]
        print(f"[Pipeline] Predicted chapter: {predicted_chapter} (Not used as filter)")

        # ── Confidence gate: reject if BERT is too uncertain ──
        if classification["chapter_confidence"] < 0.30:
            print(f"[Pipeline] ⚠️ Low confidence ({classification['chapter_confidence']:.2%}) — rejecting question.")
            return {
                "question"       : question,
                "classification" : None,
                "answer"         : "❌ Unable to process this question. The model confidence is too low — this may not be a valid NEET Biology question or it falls outside the trained syllabus.",
                "justification"  : [],
                "sources"        : [],
                "confidence"     : 0.0,
                "verified"       : False,
                "verifier_note"  : f"BERT confidence ({classification['chapter_confidence']:.2%}) is below the 30% threshold.",
                "rejected"       : True,
            }

    # ── Step 1: Retrieve relevant NCERT chunks ──────────────────────────────
    print("[Pipeline] Step 1: Retrieving NCERT chunks (Global DB Search)...")
    chunks = retrieve_chunks(question, chapter_filter=None, top_k=top_k)

    if not chunks:
        return {
            "question"       : question,
            "classification" : classification,
            "answer"         : "Insufficient context: no relevant NCERT passages found.",
            "justification"  : [],
            "sources"        : [],
            "confidence"     : 0.0,
            "verified"       : False,
            "verifier_note"  : "No chunks retrieved — cannot verify.",
        }

    # Extract top chapter/section for prompt context
    top_chunk   = chunks[0]
    top_chapter = top_chunk.get("chapter", "Unknown Chapter")
    top_section = top_chunk.get("section_title", "Unknown Topic")

    # ── Step 2: Build prompts ───────────────────────────────────────────────
    print("[Pipeline] Step 2: Building prompts...")
    answer_msgs      = build_answer_prompt(question, top_chapter, top_section, chunks)
    
    # ── Step 3: Generate answer ─────────────────────────────────────────────
    print("[Pipeline] Step 3: Generating answer via Groq...")
    answer = generate_answer(answer_msgs)

    # ── Step 4: Generate justification ──────────────────────────────────────
    print("[Pipeline] Step 4: Generating justification via Groq...")
    just_msgs     = build_justification_prompt(question, answer, chunks)
    justification = generate_justification(just_msgs)

    # ── Step 5: Verify answer (Local NLI) ───────────────────────────────────
    print("[Pipeline] Step 5: Verifying answer via NLI CrossEncoder...")
    verif_result    = verify_answer_nli(answer, chunks)
    verified        = verif_result["verified"]
    verifier_note   = verif_result["explanation"]

    # ── Step 6: Compute confidence ──────────────────────────────────────────
    retrieval_scores = [c.get("similarity", 0.0) for c in chunks]
    confidence       = compute_confidence(retrieval_scores, verified)

    # ── Step 7: Build sources list (no raw embeddings) ─────────────────────
    sources = [
        {
            "chapter"      : c.get("chapter"),
            "section"      : c.get("section"),
            "section_title": c.get("section_title"),
            "text_content" : c.get("text_content"),
            "similarity"   : round(c.get("similarity", 0.0), 4),
        }
        for c in chunks
    ]

    return {
        "question"       : question,
        "classification" : classification,
        "answer"         : answer,
        "justification"  : justification,
        "sources"        : sources,
        "confidence"     : confidence,
        "verified"       : verified,
        "verifier_note"  : verifier_note,
    }


if __name__ == "__main__":
    import json, sys

    # ── Chapters currently in your Supabase DB ──────────────────────────────
    # Kebo101 → The Living World
    # Kebo109 → Biomolecules
    TEST_QUESTIONS = [
        "What is the role of cofactors in enzyme activity?",                    # Biomolecules
        "What are primary and secondary metabolites in living organisms?",      # Biomolecules
        "How does carbonic anhydrase speed up chemical reactions?",             # Biomolecules
        "What is the difference between metabolism, anabolism and catabolism?", # Biomolecules
        "What is binomial nomenclature and who proposed it?",                  # The Living World
    ]

    question = TEST_QUESTIONS[0]   # ← change index (0-4) to test different questions
    debug    = "--debug" in sys.argv  # run with: python -m rag.pipeline --debug

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")

    result = run_pipeline(question)

    if debug:
        # Show full retrieved context so you can verify what the LLM saw
        print("\n── RETRIEVED CONTEXT ──────────────────────────────────────")
        for i, src in enumerate(result["sources"], 1):
            print(f"\n[{i}] ({src['similarity']:.3f}) {src['chapter']} > {src['section_title']}")
            print(f"    {src['text_content'][:300]}...")
        print()

    print("── ANSWER ─────────────────────────────────────────────────")
    print(result["answer"])

    print("\n── JUSTIFICATION ──────────────────────────────────────────")
    for step in result["justification"]:
        print(f"  {step}")

    print(f"\n── CONFIDENCE: {result['confidence']}  |  VERIFIED: {result['verified']} ──")
    if result.get("verifier_note"):
        print(f"   Verifier note: {result['verifier_note']}")
