-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create the knowledge base table to store document chunks and embeddings
create table if not exists ncert_knowledge_chunks (
  id uuid primary key default gen_random_uuid(),
  chunk_id text not null unique,        -- the custom ID from your JSON (e.g., bio_ch1_1_0001)
  subject text not null,                -- 'Biology'
  chapter text not null,                -- 'The Living World'
  chapter_number text not null,         -- '1'
  section text,                         -- '1.1'
  section_title text,                   -- 'What is Living?'
  text_content text not null,           -- the actual raw text chunk
  metadata jsonb default '{}'::jsonb,   -- strictly for any extra metadata
  embedding vector(384),                -- BGE embeddings are 384 dimensions
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create an HNSW index to make vector similarity search fast!
-- (Uses cosine distance operator: vector_cosine_ops)
create index on ncert_knowledge_chunks using hnsw (embedding vector_cosine_ops);

-- ==========================================
-- Optional: Create a function for similarity search
-- ==========================================
create or function match_ncert_chunks (
  query_embedding vector(384),
  match_threshold float,
  match_count int,
  filter_subject text default null,
  filter_chapter text default null
)
returns table (
  id uuid,
  chunk_id text,
  chapter text,
  section_title text,
  text_content text,
  similarity float
)
language sql stable
as $$
  select
    id,
    chunk_id,
    chapter,
    section_title,
    text_content,
    1 - (embedding <=> query_embedding) as similarity
  from ncert_knowledge_chunks
  where 1 - (embedding <=> query_embedding) > match_threshold
    and (filter_subject is null or subject = filter_subject)
    and (filter_chapter is null or chapter = filter_chapter)
  order by embedding <=> query_embedding
  limit match_count;
$$;
