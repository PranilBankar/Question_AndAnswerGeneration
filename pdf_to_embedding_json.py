import re
import json
import fitz  # PyMuPDF
from transformers import AutoTokenizer
from tqdm import tqdm

# -------- CONFIG -------- #
PDF_PATH = "kebo109.pdf"
OUTPUT_JSON = "biology_chunks.json"
MAX_TOKENS = 200
SUBJECT = "Biology"
TOKENIZER_MODEL = "bert-base-uncased"

# -------- LOAD TOKENIZER -------- #
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

# -------- STEP 1: EXTRACT TEXT FROM PDF -------- #
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        text = page.get_text("text")
        full_text += text + "\n"

    return full_text


# -------- STEP 2: CLEAN TEXT -------- #
def clean_text(text):
    text = re.sub(r"Reprint \d{4}-\d{2}", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    return text


# -------- STEP 3: SPLIT INTO CHAPTERS -------- #
def split_chapters(text):
    chapter_pattern = r"CHAPTER\s+(\d+)\s+([A-Z][A-Za-z\s]+)"
    matches = list(re.finditer(chapter_pattern, text))

    chapters = []

    for i, match in enumerate(matches):
        chapter_number = match.group(1)
        chapter_title = match.group(2).strip()

        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        chapter_text = text[start:end]

        chapters.append({
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "text": chapter_text
        })

    return chapters


# -------- STEP 4: SPLIT INTO SECTIONS (9.1, 9.2 etc) -------- #
def split_sections(chapter_text):
    section_pattern = r"\n(\d+\.\d+)\s+([A-Z][A-Z\s]+)"
    matches = list(re.finditer(section_pattern, chapter_text))

    sections = []

    for i, match in enumerate(matches):
        section_number = match.group(1)
        section_title = match.group(2).title()

        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(chapter_text)

        section_text = chapter_text[start:end]

        sections.append({
            "section_number": section_number,
            "section_title": section_title,
            "text": section_text
        })

    return sections


# -------- STEP 5: TOKEN-BASED CHUNKING -------- #
def chunk_text(text, max_tokens=MAX_TOKENS):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        candidate = current_chunk + " " + sentence
        token_count = len(tokenizer(candidate)["input_ids"])

        if token_count <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# -------- MAIN PIPELINE -------- #
def process_pdf():
    print("Extracting text...")
    raw_text = extract_text_from_pdf(PDF_PATH)

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("Splitting chapters...")
    chapters = split_chapters(cleaned_text)

    all_chunks = []

    for chapter in tqdm(chapters):
        chapter_num = chapter["chapter_number"]
        chapter_title = chapter["chapter_title"]

        sections = split_sections(chapter["text"])

        for section in sections:
            section_num = section["section_number"]
            section_title = section["section_title"]

            chunks = chunk_text(section["text"])

            for idx, chunk in enumerate(chunks):
                chunk_id = f"bio_ch{chapter_num}_{section_num}_{idx:03}"

                all_chunks.append({
                    "id": chunk_id,
                    "subject": SUBJECT,
                    "chapter": chapter_title,
                    "chapter_number": chapter_num,
                    "section": section_num,
                    "section_title": section_title,
                    "text": chunk,
                    "chunk_index": idx
                })

    print(f"Generated {len(all_chunks)} chunks.")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    process_pdf()