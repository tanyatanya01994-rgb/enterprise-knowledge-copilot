# Enterprise Knowledge Intelligence Copilot

## Project Overview

An evidence-grounded assistant for an enterprise HR policy collection. It retrieves policy passages, verifies whether they support a question, applies a deterministic confidence gate, and only then asks Gemini to draft an answer with page and section citations.

## Business Problem and Domain

Policy answers are difficult to locate, easy to misread, and costly to get wrong. HR policy was selected because answers need traceability, precise qualification, and safe abstention when a document does not contain the requested fact.

## Key Features

- Hybrid semantic and BM25 retrieval combined with reciprocal-rank fusion (RRF) for both meaning and exact policy terms.
- Cross-encoder reranking, strict evidence verification, and a 45% confidence threshold.
- Document-level filtering applied to both Qdrant and BM25 retrieval.
- Bounded conversation-aware query rewriting and multi-document evidence prompts.
- Incremental PDF re-indexing that replaces a same-named document without rebuilding the full store.
- Streamlit pages for Copilot, Knowledge Base, Retrieval Intelligence, and Evaluation.

## Architecture and End-to-End RAG Workflow

Documents are extracted with PyMuPDF, cleaned, structure-aware chunked, enriched with document/page/section metadata, embedded with `all-MiniLM-L6-v2`, and stored in local Qdrant. A question is rewritten when needed, searched semantically and lexically, fused with RRF, reranked, verified, confidence-scored, and—only when sufficient evidence exists—sent with verified chunks to Gemini. The UI renders deterministic citations separately from model output.

Hybrid retrieval improves recall where either natural-language meaning or exact terms matter. Reranking prioritizes the best candidates before verification. Verification and confidence scoring prevent retrieval similarity alone from becoming an answer. See [architecture.md](docs/architecture.md) for the detailed design.

### Ingestion, Chunking, and Metadata

PDFs are extracted page by page and cleaned before structure-aware word chunking (120 words with 30-word overlap). Each chunk retains its document name/type, year, page, detected section, chunk ID, and text. Those fields are stored both in `chunks.jsonl` and the Qdrant payload, enabling reproducible citations and document-level filtering.

### Retrieval and Grounding

The embedding model is `all-MiniLM-L6-v2`; Qdrant provides semantic retrieval and BM25 provides lexical retrieval. RRF merges both rankings and a cross-encoder reranks the candidates. Gemini verifies the candidate IDs before Gemini generation can occur. A verifier API failure is represented as `verifier_unavailable`, not an unsupported answer or a 0% confidence claim.

## Safety, Multi-Document Reasoning, and Memory

The generator receives only verified chunks. Unsupported questions and verification failures fail closed with a safe abstention. For questions spanning documents, verified evidence preserves document attribution and the prompt requires the answer to distinguish sources. Streamlit supplies previous chat messages to a bounded (eight-entry) query-rewriting context; if Gemini is unavailable, a conservative local reference resolver is used.

Comparison-language detection expands the all-document candidate pool and requires verified coverage from at least two source documents. One-sided evidence causes a safe abstention instead of an invented comparison. An explicit sidebar document filter remains authoritative.

## Re-indexing

Upload a PDF in **Knowledge Base** and choose **Index Documents**. The workflow extracts, cleans, chunks, updates `data/processed/chunks.jsonl`, deletes vectors only for the same document, upserts its replacements, refreshes BM25 on the next Streamlit run, and reports document/chunk totals. Stable vector IDs prevent deleted-ID gaps from overwriting another document.

## Evaluation Methodology

`evaluation/questions.json` contains supported, unsupported, follow-up, and multi-document cases. `evaluation/run_evaluation.py` records outcomes and does not treat an API/verifier error as a successful abstention. Reported metrics must be generated from an actual run; this repository does not claim unmeasured performance figures.

Run local tests with `python -m pytest -q` and compile with `python -m compileall src app.py evaluation`. The live Gemini evaluation is intentionally not run when quota is unavailable.

## Installation and Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add your Gemini key to .env
streamlit run app.py
```

`GEMINI_API_KEY` is the only required environment variable for Gemini-dependent rewriting, verification, and generation. Retrieval can initialize without it, but the system will abstain instead of claiming verified answers.

## Project Structure

`src/` contains ingestion, chunking, embeddings, retrieval, BM25, hybrid fusion, reranking, rewriting, verification, confidence, generation, and re-indexing modules. `data/documents/` contains the policy PDFs; `data/processed/` holds chunks; `evaluation/` contains questions and results; `tests/` contains verifier safety tests; `docs/` contains architecture documentation.

## Limitations and Future Improvements

Gemini availability and document quality limit final answering. The local fallback rewrites references but deliberately does not substitute for LLM verification. Future work could add authenticated model observability, richer evaluation judgments, document versioning, and per-document access controls.

## Security Notes

Do not commit `.env`, local vector data, generated evaluation results, or credentials. Copy `.env.example` and provide your own key locally; the placeholder file contains no secret.

## Application Screenshots

### AI Copilot — Grounded Answer

The Copilot retrieves relevant evidence, verifies it, applies a confidence gate, and generates a grounded response with traceable sources.

![AI Copilot](docs/screenshots/Screenshot%202026-09-09%20031719.png)

### Knowledge Base

The Knowledge Base provides visibility into indexed enterprise documents, chunk counts, and the Qdrant vector database.

![Knowledge Base](docs/screenshots/Screenshot%202026-09-09%20031748.png)

### Document Upload

New enterprise documents can be uploaded through the Knowledge Base interface and re-indexed into the retrieval system.

![Document Upload](docs/screenshots/Screenshot%202026-09-09%20031820.png)

### Retrieval Intelligence

The Retrieval Intelligence view exposes the complete retrieval path, including query rewriting, semantic search, BM25, RRF fusion, reranking, verification, and confidence scoring.

![Retrieval Intelligence](docs/screenshots/Screenshot%202026-09-09%20031842.png)

### Verified Evidence and Citations

Retrieved evidence is displayed with document, page, and section information to make the generated answer traceable.

![Evidence and Citations](docs/screenshots/Screenshot%202026-09-09%20031910.png)

### RAG Evaluation

The evaluation interface exposes confidence, verified evidence, reranked candidates, abstention status, and evaluation dimensions.

![RAG Evaluation](docs/screenshots/Screenshot%202026-09-09%20031936.png)
