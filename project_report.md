# NEET Biology Question & Answer Generation System — Project Report

## 1. Project Objective

Build an **end-to-end AI-powered Question Answering system** for NEET Biology that:

1. Ingests NCERT Class 11 & 12 Biology textbooks (PDFs)
2. Chunks, embeds, and stores them in a **vector database**
3. Uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer student questions grounded in NCERT content
4. Fine-tunes a **BERT classifier** to predict which chapter a question belongs to — enabling targeted retrieval and routing
5. Exposes the pipeline via a **FastAPI backend** (planned)

The system is designed for NEET exam preparation, where every answer must be traceable back to specific NCERT textbook content.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│                    USER QUESTION                     │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│  BERT Chapter Classifier │  ← Classification/ (fine-tuned bert-base-uncased)
│  Predicts: chapter_id    │
└──────────┬───────────────┘
           │ chapter filter (planned integration)
           ▼
┌──────────────────────────┐     ┌─────────────────────────────┐
│   RAG Pipeline (rag/)    │────▶│  Supabase + pgvector        │
│  1. Embed question       │     │  Table: ncert_knowledge_    │
│  2. Retrieve top-k chunks│◀────│         chunks              │
│  3. Build prompt         │     │  RPC: match_ncert_chunks()  │
│  4. Generate answer (LLM)│     └─────────────────────────────┘
│  5. Justify + Verify     │
│  6. Confidence score     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   FastAPI Backend (TBD)  │  ← /qa endpoint, /ingest, /health
└──────────────────────────┘
```

---

## 3. Complete Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core runtime |
| **PDF Extraction** | PyMuPDF 1.23.8 | Extract text from NCERT PDFs |
| **Chunking** | Custom ([clean_pdf_to_json.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/pdf_to_embedding/clean_pdf_to_json.py)) | Section-aware chunking with metadata |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` (384-dim) | Local embedding generation via `sentence-transformers` |
| **Vector Database** | Supabase + pgvector | Cloud-hosted PostgreSQL with vector similarity search |
| **Vector Index** | HNSW (via pgvector) | Fast approximate nearest-neighbor search |
| **LLM Provider** | Groq API | Free, fast inference (llama-3.3-70b-versatile) |
| **Classification** | HuggingFace `bert-base-uncased` | Fine-tuned for chapter prediction |
| **Training Framework** | HuggingFace `Trainer` + PyTorch | BERT fine-tuning with early stopping |
| **Backend (planned)** | FastAPI + Uvicorn | REST API serving |
| **Data Validation** | Pydantic | Request/response schemas |
| **Environment** | python-dotenv | Secrets management via [.env](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/pdf_to_embedding/.env) |
| **Dependencies** | pip + [requirements.txt](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/requirements.txt) | All versions pinned for reproducibility |

---

## 4. Repository Structure

```
Question_AndAnswerGeneration/
├── .env                          # (gitignored) API keys
├── .gitignore                    # .venv, .env, __pycache__, *.pyc
├── requirements.txt              # All dependencies pinned
├── taxonomy_full.txt             # Complete NCERT syllabus taxonomy
│
├── pdf_to_embedding/             # 🟢 PHASE 1: Data Pipeline
│   ├── .env / .env.example       # Supabase + Groq credentials
│   ├── clean_pdf_to_json.py      # PDF → section-aware JSON chunks
│   ├── embed_and_upload.py       # JSON → embeddings → Supabase upsert
│   ├── supabase_schema.sql       # Table + RPC + HNSW index DDL
│   └── json_ouput/               # Intermediate JSON files
│       ├── 11th Ncert output/    #   19 files (Kebo101–Kebo119)
│       └── 12th Ncert Output/    #   13 files (Lebo101–Lebo113)
│
├── rag/                          # 🟢 PHASE 2: RAG Pipeline
│   ├── __init__.py
│   ├── retriever.py              # Embed question → Supabase RPC → top-k chunks
│   ├── prompt_builder.py         # Build answer/justification/verifier prompts
│   ├── generator.py              # Groq LLM calls (answer + justify + verify)
│   ├── confidence.py             # Confidence scorer (retrieval + verification)
│   └── pipeline.py               # Orchestrator with smoke test (--debug flag)
│
└── Classification/               # 🟡 PHASE 3: BERT Classifier (scaffold ready)
    ├── __init__.py
    ├── config.json               # All hyperparameters (model, lr, epochs, etc.)
    ├── dataset.py                # PyTorch Dataset + label_map builder
    ├── train.py                  # Full HuggingFace Trainer script
    ├── predict.py                # ClassifierPredictor inference wrapper
    └── data/
        ├── sample_questions.json # Placeholder labeled data (3 examples)
        └── label_map.json        # (auto-generated) chapter_id → chapter_name
```

---

## 5. What Is Completed ✅

### Phase 1 — Data Ingestion Pipeline ✅
| Component | File | Status |
|---|---|---|
| PDF text extraction | [clean_pdf_to_json.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/pdf_to_embedding/clean_pdf_to_json.py) | ✅ Complete |
| Section-aware chunking with metadata | [clean_pdf_to_json.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/pdf_to_embedding/clean_pdf_to_json.py) | ✅ Complete |
| Supabase schema + RPC + HNSW index | [supabase_schema.sql](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/pdf_to_embedding/supabase_schema.sql) | ✅ Deployed |
| Embedding generation (BGE-small) | [embed_and_upload.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/pdf_to_embedding/embed_and_upload.py) | ✅ Complete |
| Batch upsert to Supabase | [embed_and_upload.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/pdf_to_embedding/embed_and_upload.py) | ✅ Complete |
| All 32 NCERT chapters processed | `json_ouput/` | ✅ 19 (Class 11) + 13 (Class 12) |

**Data stats:** 32 JSON files → **~696 chunks** with metadata (chapter, section, section_title, chunk_text, embedding vector).

### Phase 2 — RAG Pipeline ✅
| Component | File | What It Does |
|---|---|---|
| Retriever | [retriever.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/rag/retriever.py) | Embeds query → calls `match_ncert_chunks` RPC → returns top-k relevant chunks |
| Prompt Builder | [prompt_builder.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/rag/prompt_builder.py) | Constructs 3 prompt types: answer, justification, verifier |
| Generator | [generator.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/rag/generator.py) | Calls Groq API for answer generation, step-by-step justification, and YES/NO faithfulness verification |
| Confidence Scorer | [confidence.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/rag/confidence.py) | Combines retrieval similarity + verifier output into a 0-1 confidence score |
| Pipeline Orchestrator | [pipeline.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/rag/pipeline.py) | End-to-end flow: question → retrieve → generate → justify → verify → confidence. Includes `--debug` smoke test |

**RAG design highlights:**
- **Multi-stage pipeline**: Retrieve → Answer → Justify → Verify → Confidence
- **Singleton pattern** for Supabase client, embedding model, and Groq client (no repeated initialization)
- **Similarity threshold** of 0.1 (tuned lower to avoid false negatives with limited data)
- **Groq model**: `llama-3.3-70b-versatile` — fast, free tier, high quality

### Phase 3 — BERT Classification Layer 🟡 (Scaffold Complete)
| Component | File | Status |
|---|---|---|
| Config | [config.json](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/Classification/config.json) | ✅ All hyperparameters defined |
| Dataset class | [dataset.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/Classification/dataset.py) | ✅ Tokenizer + label_map builder |
| Training script | [train.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/Classification/train.py) | ✅ Full Trainer with early stopping, F1 metric |
| Inference wrapper | [predict.py](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/Classification/predict.py) | ✅ [ClassifierPredictor](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/Classification/predict.py#26-109) with top-k + confidence |
| Labeled training data | `data/` | 🔴 Only 3 sample examples — needs real NEET questions |

**Classifier config:**
```json
{
  "model_name": "bert-base-uncased",
  "max_length": 128,
  "num_labels": 16,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "num_epochs": 10,
  "warmup_ratio": 0.1,
  "eval_strategy": "epoch",
  "metric_for_best_model": "f1",
  "early_stopping_patience": 2
}
```

---

## 6. NCERT Dataset Summary

| Class | Chapters | Files | Total Chunks | Unique Sections |
|---|---|---|---|---|
| 11th | 19 (Ch 1–19) | Kebo101–Kebo119 | ~263 | ~200+ |
| 12th | 12 (Ch 2–13) + 1 front matter | Lebo101–Lebo113 | ~433 | ~150+ |
| **Total** | **32** | **32 JSON files** | **~696** | **~350+** |

> **Note:** Class 12 Chapter 1 (Reproduction in Organisms) is missing — [Lebo101.json](file:///d:/Users/Pranil/Github%20Repos/Question_AndAnswerGeneration/pdf_to_embedding/json_ouput/12th%20Ncert%20Output/Lebo101.json) contains only front matter. Needs re-processing.

---

## 7. Environment Variables

```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbG...                    # Service Role Key
GROQ_API_KEY=gsk_...                      # Free tier API key
GROQ_MODEL=llama-3.3-70b-versatile        # Default LLM
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5    # 384-dim local embeddings
```

---

## 8. What Remains — Roadmap 🔜

### Immediate (Next Steps)

| # | Task | Priority | Effort |
|---|---|---|---|
| 1 | **Collect & label NEET questions** → map each to chapter_id | 🔴 Critical | Medium |
| 2 | **Run BERT training** (`python -m Classification.train`) | 🔴 Critical | Low (code ready) |
| 3 | **Evaluate classifier** — target >85% F1 on held-out set | 🔴 Critical | Low |
| 4 | **Fix Lebo101.json** — re-process Ch 1 Reproduction in Organisms PDF | 🟡 Medium | Low |

### Short-Term (Integration)

| # | Task | Priority |
|---|---|---|
| 5 | **Integrate classifier into RAG** — use predicted chapter to filter retrieval (pre-filter Supabase query by chapter metadata) | 🔴 Critical |
| 6 | **Build FastAPI backend** — `/qa` endpoint wrapping `pipeline.run()`, `/health`, `/ingest` | 🟡 Medium |
| 7 | **Add syllabus_taxonomy.json** — structured JSON of all chapters/sections for the classifier and frontend | 🟡 Medium |

### Medium-Term (Production Readiness)

| # | Task | Priority |
|---|---|---|
| 8 | Add unit + integration tests for RAG pipeline and classifier | 🟡 Medium |
| 9 | Build frontend UI (React / Streamlit) for question input + answer display | 🟢 Nice-to-have |
| 10 | Add question generation capability (generate practice MCQs from NCERT content) | 🟢 Future |
| 11 | Deploy to cloud (Render / Railway / Docker) | 🟢 Future |

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| **BGE-small-en-v1.5** over larger models | 384-dim is fast and good enough for NCERT-level text; runs locally without GPU |
| **Supabase + pgvector** over Pinecone/Weaviate | Free tier, PostgreSQL ecosystem, full SQL alongside vectors |
| **Groq API** over OpenAI/Anthropic | Free tier with fast inference; `llama-3.3-70b` is competitive quality |
| **bert-base-uncased** for classification | Lightweight, well-studied for text classification; fine-tunes on a single GPU |
| **Section-aware chunking** | Each chunk carries chapter/section metadata → enables filtered retrieval |
| **Multi-stage RAG** (answer → justify → verify) | Reduces hallucination by adding a verification layer before presenting answers |
| **Singleton pattern** for clients | Avoids expensive re-initialization of embedding model, Supabase client, Groq client |

---

## 10. How to Run (Quick Reference)

```bash
# 1. Activate virtual environment
.\.venv\Scripts\activate

# 2. Process a new PDF
python pdf_to_embedding/clean_pdf_to_json.py --input "path/to/chapter.pdf"

# 3. Upload embeddings to Supabase
python pdf_to_embedding/embed_and_upload.py

# 4. Test RAG pipeline (smoke test)
python -m rag.pipeline --debug

# 5. Train BERT classifier (when data is ready)
python -m Classification.train

# 6. Test classifier predictions
python -m Classification.predict
```
