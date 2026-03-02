import re
import json
import fitz
from transformers import AutoTokenizer
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

PDF_PATH = r"D:\Users\Pranil\Github Repos\NLP_pdfDataset\11th Ncert\kebo109.pdf"
OUTPUT_JSON = "biology_embedding_ready_final.json"
SUBJECT = "Biology"
MAX_TOKENS = 220
TOKENIZER_MODEL = "bert-base-uncased"

REMOVE_SUMMARY = True  # Optional toggle

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL,
    clean_up_tokenization_spaces=True
)

# ==============================
# STEP 1: Extract Raw Text
# ==============================

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


# ==============================
# STEP 2: Clean Raw Text
# ==============================

def clean_text(text):

    # Remove page numbers
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove headers
    text = re.sub(r"\nBIOLOGY\n", "\n", text, flags=re.IGNORECASE)
    # text = re.sub(r"\nBIOMOLECULES\n", "\n", text, flags=re.IGNORECASE)

    # Remove reprint line
    text = re.sub(r"Reprint.*?\n", "", text)

    # Remove weird unicode arrows
    text = re.sub(r"[←\uf8e7]+", "", text)

    # Remove everything after EXERCISES
    text = re.split(r"\nEXERCISES", text)[0]

    # Optionally remove summary
    if REMOVE_SUMMARY:
        text = re.split(r"\nSUMMARY", text)[0]

    # Normalize spacing
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


# ==============================
# STEP 3: Extract Chapter Info
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

            # Look ABOVE for true title (skip long sentences)
            for j in range(i - 1, -1, -1):
                if len(lines[j].split()) <= 5:  # short line = likely title
                    chapter_title = lines[j].title()
                    break

            break

    return chapter_number, chapter_title


# ==============================
# STEP 4: Detect Real Sections
# ==============================
def split_real_sections(text, chapter_number):

    section_pattern = rf"\n({chapter_number}\.\d+)\s*\n?([A-Za-z][^\n]+)"

    matches = list(re.finditer(section_pattern, text))

    sections = []

    for i, match in enumerate(matches):

        section_number = match.group(1)
        section_title = match.group(2).strip().title()

        start = match.end()

        # Look ahead up to next match or end
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        # ----------------------------
        # 🔥 CRITICAL FILTER
        # ----------------------------

        # Reject sidebar / TOC matches
        if len(section_text) < 500:
            continue

        # Must contain real sentences
        if section_text.count(".") < 3:
            continue

        # Must not look like menu list
        if re.search(r"\n9\.\d\n", section_text):
            continue

        sections.append({
            "section_number": section_number,
            "section_title": section_title,
            "text": section_text
        })

    return sections


# ==============================
# STEP 5: Chunking
# ==============================

def chunk_text(text):

    # Remove repeated heading inside text
    text = re.sub(r"^\d+\.\d+\s+[A-Za-z\s]+", "", text)

    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        candidate = current_chunk + " " + sentence
        token_count = len(tokenizer(candidate)["input_ids"])

        if token_count <= MAX_TOKENS:
            current_chunk = candidate
        else:
            if len(tokenizer(current_chunk)["input_ids"]) > 40:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if len(tokenizer(current_chunk)["input_ids"]) > 40:
        chunks.append(current_chunk.strip())

    return chunks


# ==============================
# MAIN PIPELINE
# ==============================

def process_pdf():

    print("Extracting text...")
    raw_text = extract_text(PDF_PATH)

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("Extracting chapter info...")
    chapter_number, chapter_title = extract_chapter_info(cleaned_text)

    print(f"Detected Chapter {chapter_number}: {chapter_title}")

    print("Detecting real sections...")
    sections = split_real_sections(cleaned_text, chapter_number)

    print(f"Valid sections detected: {len(sections)}")

    all_chunks = []
    global_index = 0

    for section in tqdm(sections):

        section_number = section["section_number"]
        section_title = section["section_title"]

        chunks = chunk_text(section["text"])

        for chunk in chunks:

            chunk_id = f"bio_ch{chapter_number}_{section_number}_{global_index:04}"

            all_chunks.append({
                "id": chunk_id,
                "subject": SUBJECT,
                "chapter": chapter_title,
                "chapter_number": chapter_number,
                "section": section_number,
                "section_title": section_title,
                "text": chunk,
            })

            global_index += 1

    print(f"\nGenerated {len(all_chunks)} clean embedding-ready chunks.")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved to {OUTPUT_JSON}")//


if __name__ == "__main__":
    process_pdf()