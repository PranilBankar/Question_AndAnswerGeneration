"""
qa.py — Question Answering API router

Endpoints:
  POST /qa       → Full RAG pipeline (classify + retrieve + answer + verify)
  POST /classify → Classification only (BERT chapter + topic extraction)
"""

import traceback
from fastapi import APIRouter, HTTPException

from Backend.schemas import (
    QuestionRequest,
    QAResponse,
    ClassifyOnlyRequest,
    ClassifyOnlyResponse,
    ClassificationResult,
    TopicInfo,
    SourceChunk,
)
from rag.pipeline import run_pipeline
from Classification.classify_and_extract import classify_question

router = APIRouter(prefix="/api", tags=["Question Answering"])


# ==============================
# POST /api/qa
# ==============================
@router.post(
    "/qa",
    response_model=QAResponse,
    summary="Answer a NEET Biology question",
    description=(
        "Runs the full pipeline: BERT classifies the chapter, "
        "retrieves relevant NCERT chunks, generates an answer with "
        "justification, and verifies correctness."
    ),
)
async def answer_question(req: QuestionRequest):
    """
    Full Q&A pipeline:
      1. BERT → chapter prediction + topic extraction
      2. Supabase → retrieve NCERT chunks (filtered by chapter)
      3. Groq LLM → generate answer + justification
      4. Groq LLM → verify answer
      5. Compute confidence score
    """
    try:
        result = run_pipeline(
            question=req.question,
            chapter_filter=req.chapter_filter,
            top_k=req.top_k,
            use_classifier=req.use_classifier,
        )

        # Build classification model if present
        classification = None
        if result.get("classification"):
            raw = result["classification"]
            classification = ClassificationResult(
                chapter=raw["chapter"],
                chapter_id=raw["chapter_id"],
                chapter_confidence=raw["chapter_confidence"],
                topics=[
                    TopicInfo(
                        section_code=t["section_code"],
                        section_title=t["section_title"],
                        similarity=t["similarity"],
                    )
                    for t in raw.get("topics", [])
                ],
            )

        # Build sources
        sources = [
            SourceChunk(
                chapter=s.get("chapter"),
                section=s.get("section"),
                section_title=s.get("section_title"),
                text_content=s.get("text_content"),
                similarity=s.get("similarity", 0.0),
            )
            for s in result.get("sources", [])
        ]

        return QAResponse(
            question=result["question"],
            classification=classification,
            answer=result["answer"],
            justification=result.get("justification", []),
            sources=sources,
            confidence=result.get("confidence", 0.0),
            verified=result.get("verified", False),
            verifier_note=result.get("verifier_note", ""),
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}",
        )


# ==============================
# POST /api/classify
# ==============================
@router.post(
    "/classify",
    response_model=ClassifyOnlyResponse,
    summary="Classify a question (chapter + topics only)",
    description=(
        "Runs only the BERT classifier + topic extraction. "
        "Does NOT generate an answer. Useful for quick categorization."
    ),
)
async def classify_only(req: ClassifyOnlyRequest):
    """
    Classification only (no RAG):
      1. BERT → chapter prediction
      2. Supabase retrieval → topic extraction
    """
    try:
        raw = classify_question(req.question)

        classification = ClassificationResult(
            chapter=raw["chapter"],
            chapter_id=raw["chapter_id"],
            chapter_confidence=raw["chapter_confidence"],
            topics=[
                TopicInfo(
                    section_code=t["section_code"],
                    section_title=t["section_title"],
                    similarity=t["similarity"],
                )
                for t in raw.get("topics", [])
            ],
        )

        return ClassifyOnlyResponse(
            question=req.question,
            classification=classification,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}",
        )
