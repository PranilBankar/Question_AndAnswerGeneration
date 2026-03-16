# 🧬 NEET Biology Q&A — AI Study Assistant

An AI-powered Question Answering system for NEET Biology built with **BERT classification**, **RAG pipeline**, and **NCERT textbook context**.

---

## 🏗️ Architecture

```
User Question
    │
    ▼
┌────────────────────────┐
│  BERT Classifier       │ → Predicts NCERT chapter (31 classes, 91.7% F1)
│  (bert-base-uncased)   │ → Extracts 1-3 relevant topics/subtopics
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Supabase pgvector     │ → Retrieves top-k text chunks filtered by chapter
│  (BGE-small-en-v1.5)   │ → Cosine similarity search on NCERT embeddings
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Groq LLM (Llama 3)   │ → Generates answer from retrieved context
│  RAG Pipeline          │ → Justifies with step-by-step reasoning
│                        │ → Verifies answer accuracy
└────────┬───────────────┘
         │
         ▼
    JSON Response → Frontend
```

---

## 📁 Project Structure

```
Question_AndAnswerGeneration/
│
├── Backend/                    # FastAPI REST API
│   ├── main.py                 # App entrypoint, CORS, model preloading
│   ├── schemas.py              # Pydantic request/response models
│   └── routers/
│       └── qa.py               # POST /api/qa, POST /api/classify
│
├── Frontend/                   # Vanilla HTML/CSS/JS UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── Classification/             # BERT Chapter Classifier
│   ├── config.json             # Training hyperparameters
│   ├── dataset.py              # PyTorch Dataset class
│   ├── train.py                # Fine-tuning script
│   ├── predict.py              # Inference wrapper
│   ├── classify_and_extract.py # BERT + topic extraction orchestrator
│   ├── checkpoints/
│   │   └── best_model/         # Trained BERT weights
│   ├── data/
│   │   ├── generated_questions.json  # 5,673 labeled questions
│   │   └── label_map.json            # chapter_id → chapter_name
│   └── Evaluation_metrics/
│       ├── eval_metrics.json
│       └── training_log.json
│
├── rag/                        # RAG Pipeline
│   ├── retriever.py            # Supabase vector search
│   ├── prompt_builder.py       # LLM prompt templates
│   ├── generator.py            # Groq API calls
│   ├── confidence.py           # Confidence scoring
│   └── pipeline.py             # Full orchestrator
│
└── pdf_to_embedding/           # Data Ingestion
    ├── clean_pdf_to_json.py    # PDF → cleaned JSON chunks
    ├── embed_and_upload.py     # JSON → embeddings → Supabase
    └── supabase_schema.sql     # Database schema + RPC function
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Supabase account (with pgvector enabled)
- Groq API key

### 1. Clone & Setup

```bash
git clone https://github.com/PranilBankar/Question_AndAnswerGeneration.git
cd Question_AndAnswerGeneration
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch transformers sentence-transformers scikit-learn
pip install fastapi uvicorn
pip install supabase python-dotenv
pip install groq numpy
```

### 4. Environment Variables

Create `pdf_to_embedding/.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
GROQ_API_KEY=your-groq-api-key
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

---

## ▶️ Running the Project

### Start the Backend (Terminal 1)

```bash
cd Question_AndAnswerGeneration
.venv\Scripts\Activate.ps1
uvicorn Backend.main:app --reload --port 8000
```

Wait for: `✅ All models loaded — server ready!`

### Start the Frontend (Terminal 2)

```bash
cd Question_AndAnswerGeneration
python -m http.server 5500 --directory Frontend
```

### Open in Browser

```
Frontend:  http://localhost:5500
Swagger:   http://localhost:8000/docs
Health:    http://localhost:8000/health
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/qa` | Full pipeline: classify → retrieve → answer → verify |
| `POST` | `/api/classify` | BERT classification + topic extraction only |
| `GET`  | `/health` | Health check (server + model status) |
| `GET`  | `/docs` | Swagger UI (auto-generated) |

### Example: POST /api/qa

```bash
curl -X POST http://localhost:8000/api/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the role of cofactors in enzyme activity?"}'
```

### Example: POST /api/classify

```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the role of cofactors in enzyme activity?"}'
```

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|------------|
| Classifier | BERT (bert-base-uncased), PyTorch, HuggingFace Transformers |
| Embeddings | BGE-small-en-v1.5 (384-dim), Sentence Transformers |
| Vector DB | Supabase PostgreSQL + pgvector (HNSW index) |
| LLM | Groq API (Llama 3) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Data Source | NCERT Biology Class 11 & 12 (31 chapters) |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 91.19% |
| F1 (macro) | 91.68% |
| Classes | 31 chapters |
| Training Data | 5,673 questions |
| Training Time | 8 min 30 sec (CPU) |

---

## 📝 License

This project is for educational purposes.
