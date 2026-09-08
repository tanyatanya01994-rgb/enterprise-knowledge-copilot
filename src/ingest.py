from pathlib import Path
import fitz


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# CLEAN EXTRACTED TEXT
# ============================================================

def clean_text(text):
    """
    Clean extracted PDF text while preserving
    meaningful policy content.
    """

    # Remove null characters
    text = text.replace("\x00", " ")

    # Remove unwanted dataset/header/footer text
    unwanted_patterns = [
        "NexaCore Technologies | Public Training Dataset",
        "Public Training Dataset",
    ]

    for pattern in unwanted_patterns:
        text = text.replace(pattern, "")

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


# ============================================================
# EXTRACT PDF PAGE BY PAGE
# ============================================================

def extract_pdf(pdf_path):
    """
    Extract text from a PDF page by page.

    Page numbers are preserved so that the RAG system
    can provide accurate source attribution.
    """

    pages = []

    document = fitz.open(pdf_path)

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text()

        text = clean_text(text)

        if text:

            pages.append(
                {
                    "page_number": page_number,
                    "text": text
                }
            )

    document.close()

    return pages


# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

def main():

    # Create processed directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find all PDFs
    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print("No PDF files found.")

        return

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    # --------------------------------------------------------
    # Process every PDF
    # --------------------------------------------------------

    for pdf_path in pdf_files:

        print(
            f"\nProcessing: {pdf_path.name}"
        )

        try:

            pages = extract_pdf(
                pdf_path
            )

            # Output filename
            output_file = (
                PROCESSED_DIR
                / f"{pdf_path.stem}.txt"
            )

            # Write cleaned text
            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as file:

                for page in pages:

                    file.write(
                        f"--- Page "
                        f"{page['page_number']} "
                        f"---\n"
                    )

                    file.write(
                        page["text"]
                    )

                    file.write(
                        "\n\n"
                    )

            print(
                f"Saved: {output_file.name}"
            )

            print(
                f"Pages extracted: {len(pages)}"
            )

        except Exception as error:

            print(
                f"ERROR processing "
                f"{pdf_path.name}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    print("\n" + "=" * 60)

    print(
        "DOCUMENT INGESTION COMPLETED."
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()