from __future__ import annotations

import contextlib
import gc
import io
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import CONFIDENCE_THRESHOLD, initialize_retrieval, retrieve_evidence
from answer_generator import generate_answer
from hybrid_search import get_available_documents
from reindexer import reindex_document

st.set_page_config(
    page_title="Enterprise Knowledge Copilot",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- THEME -------------------------
st.markdown("""
<style>
.stApp{
 background:
 radial-gradient(circle at 85% 5%,rgba(65,126,255,.20),transparent 28%),
 radial-gradient(circle at 65% 45%,rgba(137,83,255,.14),transparent 30%),
 radial-gradient(circle at 5% 75%,rgba(36,210,178,.08),transparent 25%),
 linear-gradient(135deg,#071426,#050b17 72%);
 color:#eef5ff;
}
[data-testid="stHeader"]{background:rgba(4,10,20,.72)}
.block-container{max-width:1500px;padding-top:1.4rem;padding-bottom:5rem}
section[data-testid="stSidebar"]{
 background:linear-gradient(180deg,#08172b,#06101e);
 border-right:1px solid rgba(76,137,255,.22);
}
h1,h2,h3,h4{color:#f4f8ff!important}
p,li,label{color:#b8c9df}
.hero{
 padding:1.35rem 1.5rem 1.2rem;border-radius:22px;
 border:1px solid rgba(76,137,255,.28);
 background:linear-gradient(135deg,rgba(18,43,82,.9),rgba(10,23,44,.82));
 box-shadow:0 18px 50px rgba(0,0,0,.22);margin-bottom:1rem
}
.eyebrow{color:#45d8ff;font-size:.74rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}
.hero-title{
 font-size:2.45rem;font-weight:850;letter-spacing:-.045em;line-height:1.05;
 background:linear-gradient(90deg,#fff,#9cc8ff 48%,#b89aff);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent
}
.hero-subtitle{color:#91a9c8;margin-top:.55rem;font-size:1rem}
.brand-title{font-size:1.15rem;font-weight:850;letter-spacing:.09em;color:#fff}
.brand-subtitle{color:#7693b8;font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;margin-top:.25rem}
.section-label{color:#6f89ae;font-size:.7rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin:.9rem 0 .45rem}
.status{
 display:inline-flex;align-items:center;gap:.45rem;padding:.42rem .7rem;border-radius:999px;
 border:1px solid rgba(52,214,177,.35);background:rgba(52,214,177,.08);
 color:#67e6c5;font-size:.72rem;font-weight:800
}
.dot{width:8px;height:8px;border-radius:50%;background:#34d6b1;box-shadow:0 0 12px #34d6b1}
.feature-card,.answer-panel,.pipeline-step{
 border:1px solid rgba(76,137,255,.20);border-radius:17px;
 background:linear-gradient(145deg,rgba(16,38,72,.88),rgba(8,21,40,.94));
 box-shadow:0 12px 35px rgba(0,0,0,.16)
}
.feature-card{padding:1rem;min-height:120px}
.feature-value{font-size:1.25rem;font-weight:850;color:#fff}
.feature-name{color:#9bb2d0;font-size:.78rem;margin-top:.18rem}
.feature-note{color:#65bfff;font-size:.68rem;margin-top:.35rem}
.answer-panel{padding:1.25rem;min-height:210px}
.answer-heading{color:#9ec7ff;font-size:.76rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
.empty-answer{text-align:center;padding:2.6rem 1rem;color:#7892b5}
.mini-title{color:#fff;font-size:1.05rem;font-weight:800;margin:.35rem 0}
.muted{color:#7892b5}
.pipeline-step{padding:.85rem;min-height:110px}
.pipeline-number{color:#45d8ff;font-size:.7rem;font-weight:850}
.pipeline-title{color:#fff;font-weight:800;margin-top:.3rem}
.pipeline-desc{color:#7892b5;font-size:.72rem;margin-top:.3rem;line-height:1.4}
div[data-testid="stMetric"]{
 background:rgba(12,29,53,.78);border:1px solid rgba(76,137,255,.18);
 padding:.8rem 1rem;border-radius:16px
}
div[data-testid="stMetricLabel"]{color:#7993b6!important}
div[data-testid="stMetricValue"]{color:#f2f7ff!important}
.stButton>button{
 border-radius:12px;border:1px solid rgba(76,137,255,.27);
 background:linear-gradient(135deg,#102c56,#172b51);color:#dceaff;font-weight:700;
 min-height:2.5rem;transition:.18s
}
.stButton>button:hover{border-color:#66a9ff;color:#fff;transform:translateY(-1px)}
button[kind="primary"]{
 background:linear-gradient(135deg,#347cff,#785cff)!important;border:none!important;color:#fff!important
}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,
div[data-testid="stFileUploaderDropzone"]{
 background:#0d203b!important;border-color:#24466f!important
}
div[data-baseweb="input"] input{color:#fff!important}
.stChatMessage{background:rgba(12,29,53,.52);border:1px solid rgba(76,137,255,.13);border-radius:16px}
[data-testid="stExpander"]{border:1px solid rgba(76,137,255,.16);border-radius:15px;background:rgba(10,25,47,.58)}
.footer{text-align:center;color:#5f789c;font-size:.68rem;padding:1.4rem 0 .4rem}
</style>
""", unsafe_allow_html=True)

# ------------------------- STATE -------------------------
for key, default in {
    "messages": [],
    "last_result": None,
    "active_page": "Copilot",
    "pending_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource(show_spinner=False)
def load_backend():
    return initialize_retrieval()

try:
    with st.spinner("Starting the knowledge engine..."):
        chunks, embedding_model, qdrant_client, bm25 = load_backend()
    backend_ready = True
    backend_error = None
except Exception as error:
    backend_ready = False
    backend_error = error
    chunks, embedding_model, qdrant_client, bm25 = [], None, None, None

documents = get_available_documents(chunks) if backend_ready else []

# ------------------------- SIDEBAR -------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">KNOWLEDGE AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Enterprise Knowledge Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Platform</div>', unsafe_allow_html=True)

    for label, icon in [
        ("Copilot","AI"),("Knowledge Base","KB"),
        ("Retrieval Intelligence","RI"),("Evaluation","EV")
    ]:
        active = st.session_state.active_page == label
        if st.button(
            f"{icon}  {label}", key=f"nav_{label}", use_container_width=True,
            type="primary" if active else "secondary"
        ):
            st.session_state.active_page = label
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)

    if backend_ready:
        st.markdown('<div class="status"><span class="dot"></span> KNOWLEDGE ENGINE READY</div>', unsafe_allow_html=True)
        st.metric("Indexed Chunks", len(chunks))
        st.metric("Documents", len(documents))
    else:
        st.error("Knowledge engine could not start.")
        with st.expander("Technical error"):
            st.code(str(backend_error))

    st.markdown("---")
    st.markdown('<div class="section-label">Document Scope</div>', unsafe_allow_html=True)
    if backend_ready:
        scope = st.selectbox("Search scope", ["All Documents"] + documents, label_visibility="collapsed")
        active_document = None if scope == "All Documents" else scope
    else:
        active_document = None

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

# ------------------------- HEADER -------------------------
left, right = st.columns([6,1])
with left:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Enterprise Intelligence Platform</div>
      <div class="hero-title">Enterprise Knowledge Copilot</div>
      <div class="hero-subtitle">
        Evidence-grounded intelligence across your enterprise document collection.
      </div>
    </div>
    """, unsafe_allow_html=True)
with right:
    st.markdown('<div style="text-align:right;padding-top:18px"><div class="status"><span class="dot"></span> ONLINE</div></div>', unsafe_allow_html=True)

# ============================================================
# COPILOT
# ============================================================
if st.session_state.active_page == "Copilot":
    st.markdown("### Knowledge Overview")
    a,b,c,d = st.columns(4)
    with a: st.metric("Documents", len(documents))
    with b: st.metric("Indexed Chunks", len(chunks))
    with c: st.metric("Retrieval", "Hybrid RRF")
    with d: st.metric("Confidence Gate", f"{CONFIDENCE_THRESHOLD:.0f}%")

    st.markdown("### Ask the Knowledge Copilot")
    quick = [
        ("Annual Leave","How many annual leave days do eligible full-time employees receive?"),
        ("Work From Home","What is the company's work from home policy?"),
        ("Benefits","What employee benefits are provided?"),
        ("Exit Process","What is the resignation and exit process?"),
        ("Code of Conduct","What are the key rules in the code of conduct?"),
    ]
    cols = st.columns(5)
    for i,(label,q) in enumerate(quick):
        with cols[i]:
            if st.button(label,key=f"quick_{i}",use_container_width=True):
                st.session_state.pending_question = q

    question = st.chat_input("Ask about leave, benefits, attendance, WFH, conduct, compensation, or exit...")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages and not question:
        st.markdown("""
        <div class="answer-panel">
          <div class="empty-answer">
            <div style="font-size:2rem">AI</div>
            <div class="mini-title">Your enterprise knowledge, one question away.</div>
            <div class="muted">
              Retrieve relevant evidence, verify it, apply a confidence gate,
              and generate a grounded answer with traceable sources.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        x,y,z = st.columns(3)
        cards = [
            ("HYBRID SEARCH","Semantic + BM25","Better recall for policy language"),
            ("VERIFY EVIDENCE","Evidence verification","Only verified chunks reach the LLM"),
            ("TRACE SOURCES","Document + page + section","Transparent enterprise AI"),
        ]
        for col,(v,note,detail) in zip((x,y,z),cards):
            with col:
                st.markdown(f"""
                <div class="feature-card">
                  <div class="feature-value">{v}</div>
                  <div class="feature-name">{note}</div>
                  <div class="feature-note">{detail}</div>
                </div>
                """, unsafe_allow_html=True)

    if question and backend_ready:
        question = question.strip()
        if not question:
            st.warning("Please enter a question.")
            st.stop()

        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.markdown(question)

        history = [
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            status = st.status("Running hybrid retrieval and evidence verification...", expanded=False)
            try:
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    rewritten_query, verified_results, abstained, confidence = retrieve_evidence(
                        question, history, chunks, embedding_model, qdrant_client, bm25, active_document
                    )

                status.update(label="Retrieval and verification completed", state="complete", expanded=False)

                r1,r2,r3,r4 = st.columns(4)
                with r1: st.metric("Confidence", f"{confidence['score']:.1f}%")
                with r2: st.metric("Confidence Level", confidence["level"])
                with r3: st.metric("Verified Evidence", confidence["verified_count"])
                with r4: st.metric("Reranked Candidates", confidence["reranked_count"])
                st.progress(min(max(confidence["score"]/100,0),1))

                with st.expander("Query Intelligence"):
                    st.write("Original question")
                    st.code(question)
                    st.write("Rewritten retrieval query")
                    st.code(rewritten_query)

                if abstained:
                    answer = "The information is not available in the provided documents with sufficient confidence."
                    st.warning("The knowledge base does not contain sufficient verified evidence to answer this question safely.")
                    st.markdown(f"""
                    <div class="answer-panel">
                      <div class="answer-heading">Safe Abstention</div>
                      <div class="mini-title">{answer}</div>
                      <div class="muted">No unsupported answer was generated.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    answer = generate_answer(rewritten_query, verified_results)
                    st.markdown('<div class="answer-heading">Verified AI Answer</div>', unsafe_allow_html=True)
                    st.markdown(answer)

                    st.markdown("#### Sources & Citations")
                    seen = set()
                    for i,result in enumerate(verified_results,1):
                        chunk = result.get("chunk",{})
                        source = (
                            chunk.get("document_name","Unknown document"),
                            chunk.get("page_number","?"),
                            chunk.get("section","Unknown section")
                        )
                        if source in seen:
                            continue
                        seen.add(source)
                        with st.container(border=True):
                            st.markdown(f"**{i}. {source[0]}**")
                            st.caption(f"Page {source[1]}  |  Section: {source[2]}")

                    with st.expander(f"Evidence Intelligence · {len(verified_results)} verified chunks"):
                        for i,result in enumerate(verified_results,1):
                            chunk = result.get("chunk",{})
                            st.markdown(f"**Evidence #{i}**")
                            st.caption(
                                f"{chunk.get('document_name','Unknown')} · "
                                f"Page {chunk.get('page_number','?')} · "
                                f"{chunk.get('section','Unknown section')}"
                            )
                            st.write(chunk.get("text",""))
                            st.divider()

                st.session_state.messages.append({"role":"assistant","content":answer})
                st.session_state.last_result = {
                    "question":question,
                    "rewritten_query":rewritten_query,
                    "results":verified_results,
                    "confidence":confidence,
                    "abstained":abstained,
                }

            except Exception as error:
                status.update(label="Pipeline error",state="error",expanded=False)
                st.error("The Copilot encountered an error while processing the request.")
                with st.expander("Technical diagnostics"):
                    st.code(str(error))

# ============================================================
# KNOWLEDGE BASE
# ============================================================
elif st.session_state.active_page == "Knowledge Base":
    st.markdown("### Knowledge Base")
    st.caption("Explore the indexed enterprise document collection and retrieval footprint.")

    if not backend_ready:
        st.error("The knowledge engine is not available.")
    else:
        a,b,c = st.columns(3)
        with a: st.metric("Documents Indexed",len(documents))
        with b: st.metric("Knowledge Chunks",len(chunks))
        with c: st.metric("Vector Database","Qdrant")

        st.markdown("### Indexed Documents")
        for document in documents:
            dc = [x for x in chunks if x.get("document_name")==document]
            pages = sorted({x.get("page_number") for x in dc if x.get("page_number") is not None})
            with st.expander(document):
                x,y,z = st.columns(3)
                with x: st.metric("Chunks",len(dc))
                with y: st.metric("Pages",len(pages))
                with z: st.metric("Status","Indexed")
                st.caption("Pages: " + ", ".join(map(str,pages)))

        st.markdown("### Add Documents")
        st.info(
            "Upload PDF documents to add new knowledge to the enterprise index. "
            "Documents are processed, chunked, embedded, and indexed automatically."
        )

        files = st.file_uploader(
            "Upload enterprise documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="enterprise_document_uploader",
        )

        if files:
            st.markdown("#### Selected Documents")

            for uploaded_file in files:
                st.write(f"📄 {uploaded_file.name}")

            if st.button(
                "Index Documents",
                type="primary",
                use_container_width=True,
                key="index_uploaded_documents",
            ):
                upload_dir = PROJECT_ROOT / "data" / "documents"
                upload_dir.mkdir(parents=True, exist_ok=True)

                successful = []
                failed = []

                progress = st.progress(0)
                status_box = st.empty()

                total_files = len(files)

                # Release the cached local-Qdrant client before the
                # incremental re-indexer opens the same storage folder.
                try:
                    if qdrant_client is not None:
                        qdrant_client.close()
                except Exception:
                    pass

                load_backend.clear()
                gc.collect()

                for index, uploaded_file in enumerate(files, start=1):
                    status_box.info(
                        f"Processing {uploaded_file.name} "
                        f"({index}/{total_files})..."
                    )

                    destination = upload_dir / uploaded_file.name

                    try:
                        destination.write_bytes(uploaded_file.getbuffer())

                        result = reindex_document(destination)
                        successful.append(result)

                    except Exception as error:
                        failed.append(
                            {
                                "name": uploaded_file.name,
                                "error": str(error),
                            }
                        )

                    progress.progress(index / total_files)

                status_box.empty()

                if successful:
                    st.success(
                        f"Successfully indexed {len(successful)} document(s)."
                    )

                    for result in successful:
                        st.write(
                            f"✅ **{result['document_name']}** — "
                            f"{result['chunks_indexed']} chunks indexed"
                        )

                    total_chunks = successful[-1].get("total_chunks")
                    if total_chunks is not None:
                        st.info(
                            f"Knowledge base now contains "
                            f"**{total_chunks} indexed chunks**."
                        )

                if failed:
                    st.error(
                        f"{len(failed)} document(s) could not be indexed."
                    )

                    for item in failed:
                        with st.expander(f"❌ {item['name']}"):
                            st.code(item["error"])

                load_backend.clear()
                gc.collect()
                st.rerun()

# ============================================================
# RETRIEVAL INTELLIGENCE
# ============================================================
elif st.session_state.active_page == "Retrieval Intelligence":
    st.markdown("### Retrieval Intelligence")
    st.caption("Transparent view of the path from question to verified evidence.")

    steps = [
        ("01","Query Rewrite","Standalone retrieval query"),
        ("02","Semantic Search","Embedding + Qdrant search"),
        ("03","BM25","Lexical policy matching"),
        ("04","RRF Fusion","Semantic + keyword fusion"),
        ("05","Reranking","Cross-encoder relevance"),
        ("06","Verification","Evidence filtering"),
        ("07","Confidence Gate","Safe abstention"),
        ("08","Grounded LLM","Verified-context generation"),
    ]
    for start in range(0,8,4):
        cols=st.columns(4)
        for col,(num,title,desc) in zip(cols,steps[start:start+4]):
            with col:
                st.markdown(f"""
                <div class="pipeline-step">
                  <div class="pipeline-number">{num}</div>
                  <div class="pipeline-title">{title}</div>
                  <div class="pipeline-desc">{desc}</div>
                </div>
                """,unsafe_allow_html=True)

    result=st.session_state.last_result
    if result:
        st.markdown("#### Latest Retrieval Run")
        conf=result["confidence"]
        a,b,c,d=st.columns(4)
        with a: st.metric("Confidence",f"{conf['score']:.1f}%")
        with b: st.metric("Level",conf["level"])
        with c: st.metric("Verified",conf["verified_count"])
        with d: st.metric("Reranked",conf["reranked_count"])
        st.progress(min(max(conf["score"]/100,0),1))

        with st.container(border=True):
            st.write("Original question")
            st.code(result["question"])
            st.write("Rewritten query")
            st.code(result["rewritten_query"])

        st.markdown("#### Verified Evidence")
        for i,item in enumerate(result["results"],1):
            chunk=item.get("chunk",{})
            with st.container(border=True):
                st.markdown(f"**Evidence {i}**")
                st.caption(
                    f"{chunk.get('document_name','Unknown')} · "
                    f"Page {chunk.get('page_number','?')} · "
                    f"{chunk.get('section','Unknown section')}"
                )
                st.write(chunk.get("text",""))
    else:
        st.info("Ask a question in Copilot to populate retrieval intelligence.")

# ============================================================
# EVALUATION
# ============================================================
elif st.session_state.active_page == "Evaluation":
    st.markdown("### RAG Evaluation")
    st.caption("Current-run diagnostics for retrieval quality, grounding, and answer safety.")

    result=st.session_state.last_result
    if result:
        conf=result["confidence"]
        a,b,c,d=st.columns(4)
        with a: st.metric("Confidence",f"{conf['score']:.1f}%")
        with b: st.metric("Verified Evidence",conf["verified_count"])
        with c: st.metric("Reranked Candidates",conf["reranked_count"])
        with d: st.metric("Abstained","Yes" if result["abstained"] else "No")

        st.markdown("#### Current Run Diagnostics")
        st.dataframe({
            "Metric":[
                "Confidence Score","Confidence Threshold",
                "Verified Evidence","Reranked Candidates","Safe Abstention"
            ],
            "Value":[
                f"{conf['score']:.1f}%",f"{CONFIDENCE_THRESHOLD:.1f}%",
                conf["verified_count"],conf["reranked_count"],
                "Yes" if result["abstained"] else "No"
            ]
        },use_container_width=True,hide_index=True)

        st.markdown("#### Evaluation Dimensions")
        for title,desc in [
            ("Retrieval relevance","Were useful chunks retrieved?"),
            ("Context relevance","Did retained evidence address the query?"),
            ("Answer correctness","Does the answer match the evidence?"),
            ("Groundedness","Is the answer supported by evidence?"),
            ("Completeness","Does it cover supported parts of the question?"),
            ("Hallucination prevention","Was unsupported information avoided?"),
            ("Safe abstention","Did the system refuse when evidence was insufficient?"),
        ]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)
    else:
        st.info("Run at least one question from Copilot to populate evaluation diagnostics.")

st.markdown(
    '<div class="footer">Enterprise Knowledge Copilot · Hybrid RAG · Evidence Verification · Confidence-Aware AI</div>',
    unsafe_allow_html=True,
)
