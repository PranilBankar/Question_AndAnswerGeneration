import os
import json
import glob
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
JSON_OUTPUT_DIR = r"D:\Users\Pranil\Github Repos\Question_AndAnswerGeneration\pdf_to_embedding"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Batch size for Supabase insertion to avoid payload too large errors
BATCH_SIZE = 100

# ==============================
# INITIALIZATION
# ==============================
# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Embedding Model
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def process_and_upload():
    # Find all JSON files recursively in the output directory
    json_files = glob.glob(os.path.join(JSON_OUTPUT_DIR, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {JSON_OUTPUT_DIR}")
        return

    print(f"Found {len(json_files)} JSON files to process.")

    for json_path in json_files:
        print(f"\nProcessing file: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                chunks = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {json_path}. Skipping.")
                continue
        
        if not chunks:
            continue

        print(f"Loaded {len(chunks)} chunks.")

        # Process in batches
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Uploading Batches"):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            
            # Extract texts for embedding
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]
            
            # Generate embeddings
            embeddings = model.encode(texts_to_embed, normalize_embeddings=True)
            
            # Prepare data for Supabase
            supabase_payload = []
            for j, chunk in enumerate(batch_chunks):
                supabase_payload.append({
                    "chunk_id": chunk.get("id"),
                    "subject": chunk.get("subject", "Biology"),
                    "chapter": chunk.get("chapter", "Unknown"),
                    "chapter_number": str(chunk.get("chapter_number", "")),
                    "section": str(chunk.get("section", "")),
                    "section_title": chunk.get("section_title", ""),
                    "text_content": chunk.get("text", ""),
                    "embedding": embeddings[j].tolist()  # Convert NumPy array to list
                })
            
            # Insert into Supabase
            try:
                # Upsert is safer in case of rerun (based on chunk_id if we specify it as unique, but we let uuid handle primary id)
                # However, chunk_id is marked as unique in schema, so we should handle conflicts or just insert
                response = supabase.table("ncert_knowledge_chunks").upsert(
                    supabase_payload, 
                    on_conflict="chunk_id"
                ).execute()
                
            except Exception as e:
                print(f"Error inserting batch into Supabase: {e}")

    print("\n✅ All embeddings uploaded successfully!")


if __name__ == "__main__":
    process_and_upload()
