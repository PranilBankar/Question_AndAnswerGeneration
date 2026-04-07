"""
schemas.py — Pydantic models for FastAPI request/response validation

Defines the complete API contract for the NEET Biology Q&A system.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ==============================
# REQUEST MODELS
# ==============================

class QuestionRequest(BaseModel):
    """Request body for the /qa endpoint."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The NEET Biology question to answer.",
        examples=["What is the role of cofactors in enzyme activity?"],
    )
    chapter_filter: Optional[str] = Field(
        default=None,
        description="Optional: manually specify a chapter to filter retrieval. "
                    "If omitted, BERT classifier auto-detects the chapter.",
        examples=["Biomolecules"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of NCERT chunks to retrieve for context.",
    )
    use_classifier: bool = Field(
        default=True,
        description="Whether to use the BERT classifier for auto chapter detection.",
    )


# ==============================
# RESPONSE MODELS
# ==============================

class TopicInfo(BaseModel):
    """A predicted topic/subtopic from the NCERT syllabus."""
    section_code: str = Field(..., description="Section code (e.g., '9.8')")
    section_title: str = Field(..., description="Section title (e.g., 'Enzymes')")
    similarity: float = Field(..., description="Cosine similarity score (0-1)")


class ClassificationResult(BaseModel):
    """BERT classification output for the question."""
    chapter: str = Field(..., description="Predicted NCERT chapter name")
    chapter_id: int = Field(..., description="Chapter ID (0-30)")
    chapter_confidence: float = Field(..., description="BERT confidence score (0-1)")
    topics: list[TopicInfo] = Field(
        default=[],
        description="Top 1-3 relevant topics/subtopics within the chapter",
    )
    rejected: bool = Field(default=False, description="True if rejected by OOD gating")
    rejection_reason: str = Field(default="", description="Reason for rejection")
    entropy: float = Field(default=0.0, description="Normalized entropy of prediction")
    margin: float = Field(default=0.0, description="Confidence margin between top-1 and top-2")


class IsNeetBioRequest(BaseModel):
    """Request body for the /is-neet-bio validation endpoint."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to validate."
    )

class IsNeetBioResponse(BaseModel):
    """Response indicating whether the query is valid NEET biology."""
    is_valid: bool
    rejected: bool
    reason: str


class SourceChunk(BaseModel):
    """A retrieved NCERT text chunk used as context."""
    chapter: Optional[str] = None
    section: Optional[str] = None
    section_title: Optional[str] = None
    text_content: Optional[str] = None
    similarity: float = 0.0


class QAResponse(BaseModel):
    """Full response from the Q&A pipeline."""
    question: str = Field(..., description="The original question")
    classification: Optional[ClassificationResult] = Field(
        default=None,
        description="BERT classification result (chapter + topics)",
    )
    answer: str = Field(..., description="Generated answer from NCERT context")
    justification: list[str] = Field(
        default=[],
        description="Step-by-step justification of the answer",
    )
    sources: list[SourceChunk] = Field(
        default=[],
        description="Retrieved NCERT chunks used as evidence",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0-1)",
    )
    verified: bool = Field(..., description="Whether the answer passed verification")
    verifier_note: str = Field(
        default="",
        description="Explanation from the verifier",
    )


class ClassifyOnlyRequest(BaseModel):
    """Request body for the /classify endpoint (no RAG, just classification)."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The NEET Biology question to classify.",
        examples=["What is the role of mitochondria in ATP synthesis?"],
    )


class ClassifyOnlyResponse(BaseModel):
    """Response from the /classify endpoint."""
    question: str
    classification: ClassificationResult


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.0.0"
    components: dict = {}
