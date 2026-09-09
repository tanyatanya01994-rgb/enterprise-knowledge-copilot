System Architecture

Enterprise Knowledge Copilot

The system follows an evidence-first Retrieval-Augmented Generation (RAG) architecture. Documents are processed into structured chunks, indexed in Qdrant, retrieved using both semantic and lexical search, fused and reranked, verified, scored for confidence, and only then passed to the LLM for grounded answer generation.

flowchart LR
     ---------------- DOCUMENT INGESTION ----------------
    A[Enterprise Documents<br/>PDF / DOCX]
    B[Ingestion & Extraction<br/>Text + page numbers]
    C[Cleaning & Structure Detection<br/>Sections + metadata]
    D[Structure-Aware Chunking<br/>Chunk size + overlap]

    A --> B --> C --> D

    ---------------- KNOWLEDGE BASE ----------------
    E[Embeddings<br/>Sentence Transformer]
    F[Qdrant Vector Store<br/>Dense Index]
    G[BM25 Index<br/>Lexical Search]

    D --> E --> F
    D --> G

     ---------------- QUERY ----------------
    Q[User Question]
    H[Conversation Context]
    R[Query Rewriting<br/>Standalone retrieval query]

    Q --> R
    H --> R

     ---------------- RETRIEVAL ----------------
    S[Semantic Retrieval<br/>Qdrant]
    L[Lexical Retrieval<br/>BM25]
    X[RRF Hybrid Fusion]
    Y[Cross-Encoder Reranking]
    V[Evidence Verification]
    K[Confidence Scoring]
    T{Confidence<br/>Threshold Met?}

    R --> S
    R --> L
    S --> X
    L --> X
    X --> Y --> V --> K --> T

     ---------------- GENERATION ----------------
    N[Grounded LLM<br/>Gemini]
    O[Final Answer<br/>+ Citations + Sources]
    P[Safe Abstention<br/>Insufficient / unverified evidence]

    T -->|Yes| N --> O
    T -->|No| P

    ---------------- APPLICATION ----------------
    UI[Streamlit Application]
    KB[Knowledge Base Management<br/>Upload + Re-index]
    EV[Evaluation Runner<br/>Retrieval + Grounding Metrics]

    UI --> Q
    UI --> KB
    UI --> EV
    KB --> B
    EV --> R

    ---------------- STYLING ----------------
    classDef source fill:#E8F4FF,stroke:#2196F3,color:#123;
    classDef process fill:#EEF7EE,stroke:#43A047,color:#123;
    classDef retrieval fill:#FFF5E6,stroke:#FB8C00,color:#123;
    classDef safety fill:#FFF0F0,stroke:#E53935,color:#123;
    classDef app fill:#F3EEFF,stroke:#7E57C2,color:#123;

    class A,Q,H source;
    class B,C,D,E,F,G process;
    class S,L,X,Y retrieval;
    class V,K,T,P safety;
    class R,N,O,UI,KB,EV app;

Pipeline Stages

1. Document Ingestion

Enterprise policy documents are extracted page-by-page. Text is cleaned while preserving useful source information such as document name, page number, and section.

2. Structure-Aware Chunking

Documents are divided into retrieval-friendly chunks. Each chunk retains metadata so retrieved evidence can be traced back to its original document and location.

3. Knowledge Base

Two complementary retrieval indexes are maintained:

Qdrant — dense vector embeddings for semantic similarity.

BM25 — lexical retrieval for exact policy terminology and keyword matching.

4. Query Intelligence

The user's question is converted into a standalone retrieval query using bounded conversation context. This helps resolve follow-up questions such as "What about interns?"

5. Hybrid Retrieval

Semantic and BM25 results are combined using Reciprocal Rank Fusion (RRF). This reduces dependence on a single retrieval method.

6. Reranking

A cross-encoder reranker evaluates the strongest hybrid candidates and produces a more relevance-focused evidence set.

7. Evidence Verification

Retrieved chunks are checked before generation. Evidence that cannot adequately support the query is filtered out. If verification is unavailable, the system fails closed rather than generating an unverified answer.

8. Confidence Gate

A deterministic confidence score considers the strength and ranking of verified evidence. If confidence does not meet the configured threshold, the system abstains.

9. Grounded Generation

Only verified evidence is supplied to the Gemini generation step. The resulting answer includes traceable source information.

10. Application and Evaluation

The Streamlit application exposes:

AI Copilot

Knowledge Base

Retrieval Intelligence

Evaluation dashboard

Document upload and re-indexing

The evaluation workflow measures retrieval and answer quality and includes supported and unsupported test questions.

Design Principles

Evidence before generation

Hybrid retrieval instead of single-method search

Reranking for precision

Explicit source attribution

Confidence-aware answering

Safe abstention when evidence is insufficient

Incremental document re-indexing

Separation of retrieval, verification, generation, and UI logic

End-to-End Flow

Documents → Extraction → Cleaning → Chunking → Embeddings/Qdrant + BM25 → Query Rewrite → Hybrid Retrieval → RRF → Reranking → Evidence Verification → Confidence Gate → Grounded LLM → Cited Answer / Safe Abstention