from pathlib import Path
import json
import re


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# Chunking configuration
CHUNK_SIZE = 120
CHUNK_OVERLAP = 30


def split_into_chunks(words, chunk_size, overlap):
    """Split words into overlapping chunks."""
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        if chunk:
            chunks.append(" ".join(chunk))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def extract_page_blocks(text):
    """Separate processed text into page blocks."""
    pattern = r"--- Page (\d+) ---\n(.*?)(?=--- Page \d+ ---|$)"

    matches = re.findall(pattern, text, re.DOTALL)

    return [
        {
            "page_number": int(page_number),
            "text": page_text.strip()
        }
        for page_number, page_text in matches
    ]


def detect_section(text):
    """Try to identify the section from the extracted page text."""

    # Look for headings such as:
    # 1. Introduction
    # 2.1 Purpose
    pattern = r"(\d+(?:\.\d+)?)\.\s+([A-Z][A-Za-z &,-]+)"

    match = re.search(pattern, text)

    if match:
        return f"{match.group(1)}. {match.group(2).strip()}"

    return "Unknown"


def process_document(txt_path):
    """Create chunks and metadata for one document."""

    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()

    pages = extract_page_blocks(text)

    document_name = txt_path.stem

    chunks = []
    chunk_number = 1

    for page in pages:

        words = page["text"].split()

        page_chunks = split_into_chunks(
            words,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for chunk_text in page_chunks:

            chunk = {
                "chunk_id": f"{document_name}_{chunk_number:04d}",
                "document_name": document_name,
                "document_type": "HR Policy",
                "year": 2026,
                "page_number": page["page_number"],
                "section": detect_section(page["text"]),
                "text": chunk_text
            }

            chunks.append(chunk)
            chunk_number += 1

    return chunks


def main():

    all_chunks = []

    txt_files = list(PROCESSED_DIR.glob("*.txt"))

    if not txt_files:
        print("No processed TXT files found.")
        return

    print(f"Found {len(txt_files)} processed documents.")

    for txt_file in txt_files:

        print(f"\nChunking: {txt_file.name}")

        document_chunks = process_document(txt_file)

        all_chunks.extend(document_chunks)

        print(f"Chunks created: {len(document_chunks)}")

    output_file = PROCESSED_DIR / "chunks.jsonl"

    with open(output_file, "w", encoding="utf-8") as file:

        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False))
            file.write("\n")

    print("\n--------------------------------")
    print("Chunking completed successfully.")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {output_file}")
    print("--------------------------------")


if __name__ == "__main__":
    main()