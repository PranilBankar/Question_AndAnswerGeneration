import re
import json
import fitz  # PyMuPDF
from transformers import AutoTokenizer
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

PDF_PATH = "kebo109.pdf"
OUTPUT_JSON = "biology_clean_chunks.json"
MAX_TOKENS = 200
SUBJECT = "Biology"
TOKENIZER_MODEL = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL,
    clean_up_tokenization_spaces=True
)

# ==============================
# STEP 1: Extract Raw Text
# ==============================

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    return full_text

# print("---- RAW START ----")
# print(extract_text_from_pdf(PDF_PATH))
# print("---- RAW END ----")

def clean_text(text):

    # Remove page numbers alone in lines
    text = re.sub(r"\n\d+\n", "\n", text)

    # Remove repeating headers
    text = re.sub(r"\nBIOLOGY\n", "\n", text, flags=re.IGNORECASE)
    # text = re.sub(r"\nBIOMOLECULES\n", "\n", text, flags=re.IGNORECASE)

    # Remove table blocks
    text = re.sub(r"TABLE\s+\d+.*?(?=\n\n)", "", text, flags=re.DOTALL)

    # Remove figure captions
    text = re.sub(r"Figure\s+\d+.*?\n", "", text)

    # Remove excessive newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Normalize spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


# ==============================
# STEP 3: Detect Chapter
# ==============================

def extract_chapter_info(text):

    lines = text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]

    chapter_number = "Unknown"
    chapter_title = "Unknown"

    for i, line in enumerate(lines):
        match = re.search(r"CHAPTER\s+(\d+)", line, re.IGNORECASE)
        if match:
            chapter_number = match.group(1)

            # Take previous line as title
            if i > 0:
                chapter_title = lines[i - 1].title()

            break

    return chapter_number, chapter_title


# ==============================
# STEP 4: Split Into Sections
# ==============================

def split_sections(text):
    section_pattern = r"\n(\d+\.\d+)\s+([A-Za-z][^\n]+)"

    matches = list(re.finditer(section_pattern, text))

    sections = []

    for i, match in enumerate(matches):
        section_number = match.group(1)
        section_title = match.group(2).replace("\n", "").strip().title()

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end]

        sections.append({
            "section_number": section_number,
            "section_title": section_title,
            "text": section_text
        })

    return sections


# ==============================
# STEP 5: Sentence-Based Chunking
# ==============================

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


# ==============================
# MAIN PIPELINE
# ==============================

def process_pdf():
    print("Extracting text...")
    raw_text = extract_text_from_pdf(PDF_PATH)

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("Extracting chapter info...")
    chapter_number, chapter_title = extract_chapter_info(cleaned_text)

    print(f"Detected Chapter {chapter_number}: {chapter_title}")

    print("Splitting sections...")
    sections = split_sections(cleaned_text)

    print(f"Detected {len(sections)} sections")

    all_chunks = []

    for section in tqdm(sections):
        section_number = section["section_number"]
        section_title = section["section_title"]

        chunks = chunk_text(section["text"])

        for idx, chunk in enumerate(chunks):
            chunk_id = f"bio_ch{chapter_number}_{section_number}_{idx:03}"

            all_chunks.append({
                "id": chunk_id,
                "subject": SUBJECT,
                "chapter": chapter_title,
                "chapter_number": chapter_number,
                "section": section_number,
                "section_title": section_title,
                "text": chunk.strip(),
                "chunk_index": idx
            })

    print(f"Generated {len(all_chunks)} clean chunks.")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    process_pdf()