import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'pdf_to_embedding', '.env'))

# ==============================
# CONFIG
# ==============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = 5                    # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3   # Minimum cosine similarity to include a chunk

# ==============================
# SINGLETON CLIENTS
# ==============================
_supabase_client: Client = None
_embedding_model: SentenceTransformer = None


def get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file.")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"[Retriever] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# ==============================
# RETRIEVAL FUNCTION
# ==============================
def retrieve_chunks(
    question: str,
    chapter_filter: str = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embeds the question and retrieves the top-k most similar NCERT chunks
    from Supabase pgvector, optionally filtered by chapter.

    Returns a list of dicts: {chunk_id, chapter, section_title, text_content, similarity}
    """
    model = get_embedding_model()
    supabase = get_supabase()

    # Embed the question
    query_vector = model.encode(question, normalize_embeddings=True).tolist()

    # Call the Postgres similarity search function
    try:
        response = supabase.rpc(
            "match_ncert_chunks",
            {
                "query_embedding": query_vector,
                "match_threshold": SIMILARITY_THRESHOLD,
                "match_count": top_k,
                "filter_chapter": chapter_filter,
            },
        ).execute()
    except Exception as e:
        print(f"[Retriever] Error calling match_ncert_chunks: {e}")
        return []

    results = response.data or []

    if not results:
        print("[Retriever] Warning: No chunks found above the similarity threshold.")

    return results


if __name__ == "__main__":
    # Quick smoke test
    question = "What is the role of mitochondria in a cell?"
    chunks = retrieve_chunks(question, top_k=3)
    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"\n[{i}] ({c['similarity']:.3f}) {c['chapter']} > {c['section_title']}")
        print(f"    {c['text_content'][:120]}...")
