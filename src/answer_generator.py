import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# MODEL FALLBACK LIST
# ============================================================

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence_text(evidence):

    evidence_text = ""

    for rank, result in enumerate(
        evidence,
        start=1
    ):

        chunk = result["chunk"]

        evidence_text += f"""
Evidence {rank}:

Document:
{chunk.get("document_name")}

Page:
{chunk.get("page_number")}

Section:
{chunk.get("section")}

Text:
{chunk.get("text")}

----------------------------------------
"""

    return evidence_text


# ============================================================
# BUILD SOURCE ATTRIBUTION
# ============================================================

def build_sources(evidence):

    sources = []

    seen = set()

    for result in evidence:

        chunk = result["chunk"]

        document = chunk.get(
            "document_name",
            "Unknown document"
        )

        page = chunk.get(
            "page_number",
            "Unknown page"
        )

        section = chunk.get(
            "section",
            "Unknown section"
        )

        source_key = (
            document,
            page,
            section
        )

        # Avoid duplicate sources
        if source_key in seen:
            continue

        seen.add(source_key)

        sources.append(
            {
                "document": document,
                "page": page,
                "section": section
            }
        )

    return sources


# ============================================================
# FORMAT SOURCE ATTRIBUTION
# ============================================================

def format_sources(sources):

    if not sources:

        return (
            "\n\n"
            "SOURCES\n"
            "--------------------------------------------------\n"
            "No source information available."
        )

    source_text = (
        "\n\n"
        "SOURCES\n"
        "--------------------------------------------------\n"
    )

    for number, source in enumerate(
        sources,
        start=1
    ):

        source_text += (
            f"{number}. "
            f"{source['document']} | "
            f"Page {source['page']} | "
            f"Section: {source['section']}\n"
        )

    return source_text


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query, evidence):

    # --------------------------------------------------------
    # Build evidence for Gemini
    # --------------------------------------------------------

    evidence_text = build_evidence_text(
        evidence
    )


    # --------------------------------------------------------
    # Build deterministic source attribution
    # --------------------------------------------------------

    sources = build_sources(
        evidence
    )


    # ========================================================
    # GROUNDED RAG PROMPT
    # ========================================================

    prompt = f"""
You are an Enterprise Knowledge Assistant.

Your task is to answer the user's question using ONLY
the evidence provided below.

IMPORTANT RULES:

1. Use only the provided evidence.

2. Do not use outside knowledge.

3. Do not invent information.

4. Do not assume information that is not present
   in the evidence.

5. If the evidence does not provide enough information,
   clearly state that the information is not available
   in the provided documents.

6. Give a concise, professional answer.

7. If multiple evidence chunks are relevant,
   combine them carefully.

8. Do not mention:
   - embeddings
   - BM25
   - vector databases
   - reranking
   - query rewriting
   - internal retrieval implementation

9. Do not create or invent source names,
   page numbers, or sections.

10. The final answer must be grounded entirely
    in the provided evidence.

11. Do not add a separate SOURCES section.
    The application will generate source attribution
    separately from the retrieved evidence.

USER QUESTION:

{query}


RETRIEVED EVIDENCE:

{evidence_text}


FINAL ANSWER:
"""


    # ========================================================
    # TRY GEMINI MODELS
    # ========================================================

    last_error = None

    for model_name in MODELS:

        print(
            f"\nTrying answer model: {model_name}"
        )

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response.text:

                answer = response.text.strip()

                print(
                    f"Answer generated using: "
                    f"{model_name}"
                )

                # ------------------------------------------------
                # Add verified source attribution
                # ------------------------------------------------

                source_text = format_sources(
                    sources
                )

                return (
                    answer
                    + source_text
                )

        except Exception as error:

            last_error = error

            print(
                f"Model {model_name} unavailable."
            )

            print(
                f"Reason: {type(error).__name__}"
            )

            continue


    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    raise RuntimeError(
        "All Gemini answer-generation models are "
        "currently unavailable. Please try again later."
        f"\nLast error: {last_error}"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_evidence = [

        {
            "chunk": {

                "document_name":
                    "Leave_Policy.pdf",

                "page_number":
                    1,

                "section":
                    "Annual Leave",

                "text":
                    "Eligible full-time employees receive "
                    "18 days of annual leave per calendar "
                    "year."

            }
        }

    ]


    question = (
        "How many annual leave days do employees get?"
    )


    answer = generate_answer(
        question,
        sample_evidence
    )


    print("\n")
    print("=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print("\n" + "=" * 70)