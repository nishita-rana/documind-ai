import os
import sys
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure the root src folder is available in system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Force reload local modules during development to prevent import cache issues
import importlib
for module_name in list(sys.modules.keys()):
    if module_name.startswith("src.") or module_name == "src":
        try:
            importlib.reload(sys.modules[module_name])
        except Exception:
            pass

# Load environments
load_dotenv()

# Set Streamlit Page Configurations
st.set_page_config(
    page_title="DocuMind AI - Intelligent Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Premium Styles (Glassmorphism + Outfit Typography)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* App-wide Font family */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Header gradients */
    .app-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }

    /* Metric Cards Glassmorphism styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.35);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 1.7rem;
        font-weight: 700;
    }
    
    /* Style native streamlit metric container as premium cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.2s ease-in-out, border-color 0.2s ease-in-out !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
    }

    div[data-testid="metric-container"] label[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Source Attributions */
    .source-header {
        font-size: 0.9rem;
        font-weight: 600;
        color: #38bdf8;
    }
    
    .source-meta {
        font-size: 0.75rem;
        color: #64748b;
    }
    
    /* System Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    
    .status-healthy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-empty {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-error {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ensure Required Directories Exist
os.makedirs("docs", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ----------------- SESSION INITIALIZATION -----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_results" not in st.session_state:
    st.session_state.retrieved_results = None

# Track Metrics and Logger setup
from src.utils.logger import setup_logger
from src.evaluation.metrics import MetricsTracker
from src.utils.helpers import format_size, export_chat_to_json
from src.utils.exporters import export_to_pdf, export_to_markdown, export_to_txt

# Setup logging
setup_logger(level=os.getenv("LOG_LEVEL", "INFO"))

if "metrics_tracker" not in st.session_state:
    st.session_state.metrics_tracker = MetricsTracker()

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = st.session_state.metrics_tracker.load_history()

# ----------------- SIDEBAR INTERFACE -----------------

st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
st.sidebar.markdown("### Configuration Panel")

# LLM Provider Selection
provider = st.sidebar.selectbox(
    "LLM Provider",
    options=["OpenAI", "Ollama (Local)"],
    index=0,
    help="Select the AI service provider. OpenAI requires an API key. Ollama runs models completely locally for free."
)

if provider == "OpenAI":
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("openai_api_key", env_key),
        placeholder="sk-...",
        help="Provide your OpenAI API Key. If left empty, the application will try to load it from your local environment variables."
    )
    if api_key:
        st.session_state.openai_api_key = api_key
    ollama_model = "llama3.2:3b"
else:
    api_key = None
    ollama_model = st.sidebar.text_input(
        "Ollama Model Name",
        value=st.session_state.get("ollama_model", "llama3.2:3b"),
        help="Make sure this model is running in Ollama (e.g. run 'ollama run llama3.2:3b' in your terminal first)."
    )
    if ollama_model:
        st.session_state.ollama_model = ollama_model

current_provider = provider.lower().replace(" (local)", "")
current_model = "gpt-4o-mini" if current_provider == "openai" else ollama_model

st.sidebar.caption(f"🌐 Active Backend: **{provider}** | Model: **{current_model}**")

# Document Ingestion Section
st.sidebar.markdown("---")
st.sidebar.markdown("### Document Indexing")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents (.md, .pdf)",
    type=["md", "pdf"],
    accept_multiple_files=True,
    help="Drag and drop Markdown or PDF files to add them to the RAG database."
)

# Retrieval Config Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### Retrieval Settings")
retrieval_strategy = st.sidebar.selectbox(
    "Retrieval Strategy",
    options=["Similarity Search", "MMR Search", "Score Threshold Search"],
    index=0,
    help="Select the vector search algorithms. MMR maximizes document diversity. Score Threshold drops weak matches."
)

top_k = st.sidebar.slider(
    "Top K Chunks",
    min_value=1,
    max_value=12,
    value=4,
    step=1,
    help="The maximum number of retrieved text chunks to feed the generation prompt."
)

score_threshold = st.sidebar.slider(
    "Relevance Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    disabled=(retrieval_strategy != "Score Threshold Search"),
    help="Filter out retrieved chunks scoring below this similarity threshold. Only used in Score Threshold Search."
)

# Advanced Chunking parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### Text Splitting Controls")
chunk_size = st.sidebar.slider(
    "Chunk Size (Characters)",
    min_value=200,
    max_value=2000,
    value=1000,
    step=100,
    help="Size of text chunks into which document files will be sliced."
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap (Characters)",
    min_value=0,
    max_value=500,
    value=200,
    step=20,
    help="Size of character overlaps between consecutive chunks to preserve contextual flow."
)

# Advanced pipelines toggles
st.sidebar.markdown("---")
st.sidebar.markdown("### RAG Enhancements")
use_rewriter = st.sidebar.checkbox(
    "Enable Query Rewriting",
    value=True,
    help="Enable rewriting vague or follow-up user questions into standalone, search-optimized queries."
)

use_compressor = st.sidebar.checkbox(
    "Context Compression",
    value=False,
    help="Enable summarization compression on retrieved chunks to trim non-relevant text, saving tokens."
)

# ----------------- DYNAMIC INITIALIZATION -----------------

current_provider = provider.lower().replace(" (local)", "")
current_model = "gpt-4o-mini" if current_provider == "openai" else ollama_model

if current_provider == "openai" and not api_key:
    st.warning("⚠️ OpenAI API Key is missing. Please provide it in the sidebar to activate DocuMind AI.")
    st.stop()

# Initialize core services inside session state
required_keys = ["embed_service", "chroma_manager", "retriever", "generator", "loader", "splitter"]
any_missing = any(k not in st.session_state for k in required_keys)

if (any_missing or 
    st.session_state.get("last_provider") != current_provider or
    st.session_state.get("last_model") != current_model or
    (current_provider == "openai" and st.session_state.get("last_api_key") != api_key)):
    
    st.session_state.last_provider = current_provider
    st.session_state.last_model = current_model
    st.session_state.last_api_key = api_key
    
    # Imports are executed on demand to verify errors
    from src.embeddings.embedding_service import OpenAIEmbeddingService
    from src.vectordb.chroma_manager import ChromaManager
    from src.retrieval.retriever import DocuMindRetriever
    from src.llm.generator import LLMGenerator
    from src.ingestion.loader import DocumentLoader
    from src.chunking.splitter import MarkdownSplitter
    
    st.session_state.embed_service = OpenAIEmbeddingService(
        api_key=api_key,
        provider=current_provider,
        model=current_model if current_provider == "ollama" else "text-embedding-3-small"
    )
    st.session_state.chroma_manager = ChromaManager(
        persist_dir="chroma_db",
        collection_name="documind_collection",
        embedding_service=st.session_state.embed_service
    )
    st.session_state.retriever = DocuMindRetriever(
        chroma_manager=st.session_state.chroma_manager,
        api_key=api_key,
        provider=current_provider,
        model=current_model,
        use_rewriter=use_rewriter,
        use_compressor=use_compressor
    )
    st.session_state.generator = LLMGenerator(
        api_key=api_key,
        provider=current_provider,
        model=current_model
    )
    st.session_state.loader = DocumentLoader(docs_dir="docs", manifest_path="chroma_db/index_manifest.json")
    st.session_state.splitter = MarkdownSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
# Keep parameters updated dynamically
if "splitter" in st.session_state:
    st.session_state.splitter.chunk_size = chunk_size
    st.session_state.splitter.chunk_overlap = chunk_overlap
if "retriever" in st.session_state:
    st.session_state.retriever.use_rewriter = use_rewriter
    st.session_state.retriever.use_compressor = use_compressor

# ----------------- INGESTION PIPELINE TRIGGER -----------------

def run_indexing(rebuild: bool = False):
    loader = st.session_state.loader
    splitter = st.session_state.splitter
    chroma = st.session_state.chroma_manager

    with st.spinner("Indexing documents, please wait..."):
        if rebuild:
            try:
                chroma.clear_database()
                loader.clear_manifest()
            except Exception as e:
                st.error(f"Failed to clear database: {e}")
                return 0, 0, 0
            
        try:
            to_index, deleted, unmodified = loader.scan_directory()
            
            # 1. Process deletions from DB
            for path in deleted:
                chroma.delete_documents_by_file(path)
                loader.remove_from_manifest(path)
                
            # 2. Process added or modified files
            total_chunks_added = 0
            for doc in to_index:
                # Delete old chunks if modifying a file
                chroma.delete_documents_by_file(doc["path"])
                
                # Split
                chunks, stats = splitter.split_document(doc)
                if chunks:
                    chroma.add_documents(chunks)
                    total_chunks_added += len(chunks)
                    loader.update_manifest(doc["path"], doc["file_hash"])
                    
            loader.save_manifest()
            return len(to_index), len(deleted), total_chunks_added
        except Exception as e:
            st.error(
                f"Error during document indexing: {e}\n\n"
                "• **If using OpenAI**: Please verify your API Key quota and billing balance.\n"
                "• **If using Ollama**: Make sure your local Ollama server is running (e.g. run `ollama run llama3.2:3b` in your terminal) and the model is pulled."
            )
            return 0, 0, 0

# Save uploaded files to disk and trigger ingestion
if uploaded_files:
    saved_paths = []
    for f in uploaded_files:
        path = os.path.join("docs", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(path)
    
    indexed, deleted, chunks = run_indexing(rebuild=False)
    if indexed > 0 or deleted > 0:
        st.sidebar.success(f"Ingestion successful! Indexed {indexed} files ({chunks} chunks). Deleted {deleted} files.")
    else:
        st.sidebar.info("No modifications detected. Database is up to date.")

# Rebuild / Delete index actions
col_act1, col_act2 = st.sidebar.columns(2)
with col_act1:
    if st.button("🔄 Rebuild Index", use_container_width=True, help="Wipes the vector store and re-chunks all docs in the directory."):
        indexed, deleted, chunks = run_indexing(rebuild=True)
        st.sidebar.success(f"Index rebuilt! Loaded {chunks} chunks.")
with col_act2:
    if st.button("🗑️ Delete Index", use_container_width=True, help="Deletes the database collection and clears all manifest history."):
        st.session_state.chroma_manager.clear_database()
        st.session_state.loader.clear_manifest()
        st.sidebar.warning("Vector database successfully cleared.")

def render_query_analytics_ui(m, index):
    st.markdown("---")
    st.markdown("##### 📊 Query Retrieval & Inference Analytics")
    
    strategy_name = m.get("strategy", "Similarity Search")
    st.caption(f"🎯 **Strategy Selected:** `{strategy_name}`")
    
    # If query was rewritten, display comparison
    original_q = m.get("original_query", "")
    rewritten_q = m.get("rewritten_query", "")
    improvement = m.get("similarity_improvement", 0.0)
    
    if original_q and rewritten_q and original_q != rewritten_q:
        st.markdown(
            f"<div style='background: rgba(56, 189, 248, 0.06); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;'>"
            f"<span style='color: #38bdf8; font-weight: 600; font-size: 0.85rem;'>🔄 Query Rewriting Active</span>"
            f"<div style='margin-top: 0.4rem; font-size: 0.9rem; color: #94a3b8;'><b>Original:</b> <i>\"{original_q}\"</i></div>"
            f"<div style='margin-top: 0.2rem; font-size: 0.9rem; color: #f8fafc;'><b>Optimized Standalone:</b> <i>\"{rewritten_q}\"</i></div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Row 1: Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval Time", f"{m['retrieval_time']:.4f}s")
    c2.metric("Generation Time", f"{m['generation_time']:.4f}s")
    c3.metric("Total Response Time", f"{m['total_time']:.4f}s")
    
    # Show delta for similarity score improvement if rewriting occurred
    if original_q and rewritten_q and original_q != rewritten_q and improvement != 0.0:
        c4.metric("Avg Similarity Score", f"{m['avg_similarity_score']:.4f}", delta=f"{improvement:+.4f}")
    else:
        c4.metric("Avg Similarity Score", f"{m['avg_similarity_score']:.4f}")
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Chunks Retrieved", f"{m['chunks_retrieved']}")
    c6.metric("Vector Store Size", f"{m['vector_store_size']} chunks")
    c7.metric("Docs Indexed", f"{m['total_documents_indexed']}")
    if original_q and rewritten_q and original_q != rewritten_q and improvement != 0.0:
        c8.metric("Retrieval Improvement", f"{improvement * 100:+.2f}%")
    else:
        c8.empty()
    
    # Collect history of times up to the current index
    history_metrics = []
    for past_msg in st.session_state.messages[:index + 1]:
        if past_msg.get("role") == "assistant" and past_msg.get("metrics"):
            history_metrics.append(past_msg["metrics"])
            
    # Row 2: Charts inside an expander
    with st.expander("📈 Visual Performance Charts", expanded=False):
        ch1, ch2 = st.columns(2)
        with ch1:
            st.caption("**Response Time History (Seconds)**")
            if history_metrics:
                history_df = pd.DataFrame({
                    "Query": [f"Q{j+1}" for j in range(len(history_metrics))],
                    "Retrieval Time": [h["retrieval_time"] for h in history_metrics],
                    "Generation Time": [h["generation_time"] for h in history_metrics],
                    "Total Time": [h["total_time"] for h in history_metrics]
                })
                st.line_chart(history_df.set_index("Query"), height=200)
            else:
                st.info("No response time history available.")
                
        with ch2:
            st.caption("**Retrieval Similarity Score Distribution**")
            scores = m.get("scores", [])
            if scores:
                score_df = pd.DataFrame({
                    "Chunk Index": [f"Chunk {i+1}" for i in range(len(scores))],
                    "Score": scores
                })
                st.bar_chart(score_df.set_index("Chunk Index"), height=200)
            else:
                st.info("No similarity scores available.")

# ----------------- MAIN VIEW LAYOUT -----------------

st.markdown("<h1 class='app-title'>🧠 DocuMind AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Intelligent Knowledge Assistant powered by RAG</p>", unsafe_allow_html=True)

# Layout main page in tabs
tab_chat, tab_sources, tab_health, tab_analytics = st.tabs([
    "💬 Chat Assistant", 
    "📑 Source Attribution", 
    "🏥 Database Health", 
    "📈 Retrieval Analytics Dashboard"
])

# ----------------- TAB 1: CHAT ASSISTANT -----------------

with tab_chat:
    # Clear conversation action
    col_chat_title, col_chat_actions = st.columns([3, 7])
    with col_chat_actions:
        col_c1, col_c2, col_c3, col_c4, col_c5 = col_chat_actions.columns(5)
        with col_c1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.retrieved_results = None
                st.rerun()
        with col_c2:
            # Export TXT button
            if st.session_state.messages:
                txt_str = export_to_txt(st.session_state.messages)
                st.download_button(
                    label="📋 Export TXT",
                    data=txt_str,
                    file_name="documind_chat_history.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.button("📋 Export TXT", disabled=True, use_container_width=True)
        with col_c3:
            # Export MD button
            if st.session_state.messages:
                md_str = export_to_markdown(st.session_state.messages)
                st.download_button(
                    label="📝 Export MD",
                    data=md_str,
                    file_name="documind_chat_history.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            else:
                st.button("📝 Export MD", disabled=True, use_container_width=True)
        with col_c4:
            # Export PDF button
            if st.session_state.messages:
                try:
                    pdf_bytes = export_to_pdf(st.session_state.messages)
                    st.download_button(
                        label="📄 Export PDF",
                        data=pdf_bytes,
                        file_name="documind_chat_history.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.button("📄 Export PDF", disabled=True, use_container_width=True)
            else:
                st.button("📄 Export PDF", disabled=True, use_container_width=True)
        with col_c5:
            # Download JSON button
            if st.session_state.messages:
                json_str = export_chat_to_json(st.session_state.messages)
                st.download_button(
                    label="📥 Export JSON",
                    data=json_str,
                    file_name="documind_chat_history.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.button("📥 Export JSON", disabled=True, use_container_width=True)

    # Render conversational transcript
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Show dedicated "Sources" section below assistant answers
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown("---")
                st.markdown("##### 📚 Sources & Attributions")
                
                # Render up to top 3 retrieved chunks
                for idx, src in enumerate(msg["sources"]):
                    score = src["score"]
                    if score >= 0.85:
                        badge = "🟢 High Confidence"
                    elif score >= 0.65:
                        badge = "🟡 Medium Confidence"
                    else:
                        badge = "🔴 Low Confidence"
                    
                    title = f"[{idx+1}] {src['filename']} — Chunk: {src['chunk_id']} ({badge})"
                    with st.expander(title):
                        c1, c2 = st.columns(2)
                        c1.metric("Similarity Score", f"{score:.4f}")
                        c2.metric("Chunk Length", f"{src['length']} chars")
                        st.markdown("**Retrieved Chunk Text:**")
                        st.code(src["content"], language="markdown")
                        
            # Show dedicated "Retrieval Analytics" section
            if msg["role"] == "assistant" and msg.get("metrics"):
                render_query_analytics_ui(msg["metrics"], msg_idx)
            
    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Render User Message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Run Retrieval Layer
        retrieval_res = st.session_state.retriever.retrieve(
            query=prompt,
            chat_history=st.session_state.messages[:-1],  # Exclude current prompt from memory calculation
            strategy=retrieval_strategy,
            k=top_k,
            score_threshold=score_threshold
        )
        
        # Save retrieval results for the Sources tab
        st.session_state.retrieved_results = retrieval_res

        # 3. Stream Assistant Response
        start_gen_time = time.time()
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            metrics = None
            
            # Request generator stream
            generator_stream = st.session_state.generator.generate_response(
                query=retrieval_res["query"],
                context_documents=retrieval_res["chunks"],
                chat_history=st.session_state.messages[:-1]
            )
            
            for item in generator_stream:
                if item["type"] == "text":
                    full_response += item["content"]
                    response_container.markdown(full_response + "▌")
                elif item["type"] == "metrics":
                    metrics = item
            
            response_container.markdown(full_response)
            
        generation_latency = time.time() - start_gen_time
        
        # Get collection stats for indexing metadata
        db_stats = st.session_state.chroma_manager.get_collection_stats()
        scores = retrieval_res.get("scores", [])
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        query_metrics = {
            "retrieval_time": retrieval_res["latency"],
            "generation_time": generation_latency,
            "total_time": retrieval_res["latency"] + generation_latency,
            "chunks_retrieved": len(retrieval_res["original_chunks"]),
            "avg_similarity_score": avg_score,
            "vector_store_size": db_stats.get("total_chunks", 0),
            "total_documents_indexed": db_stats.get("unique_files_count", 0),
            "scores": scores,
            "original_query": prompt,
            "rewritten_query": retrieval_res.get("rewritten_query", retrieval_res["query"]),
            "similarity_improvement": retrieval_res.get("similarity_improvement", 0.0),
            "strategy": retrieval_strategy
        }
        
        # Extract top 3 retrieved chunks for inline sources attribution
        sources_list = []
        if retrieval_res and "chunks" in retrieval_res:
            for idx, doc in enumerate(retrieval_res["chunks"][:3]):
                score = retrieval_res["scores"][idx] if idx < len(retrieval_res["scores"]) else 0.0
                sources_list.append({
                    "filename": doc.metadata.get("filename", "Unknown File"),
                    "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                    "score": score,
                    "length": len(doc.page_content),
                    "content": doc.page_content
                })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources_list,
            "metrics": query_metrics
        })

        # 4. Log RAG Performance Metrics
        log_entry = st.session_state.metrics_tracker.log_run(
            query=prompt,
            retrieved_count=len(retrieval_res["original_chunks"]),
            compressed_count=len(retrieval_res["chunks"]),
            retrieval_latency=retrieval_res["latency"],
            generation_latency=generation_latency,
            prompt_tokens=metrics["prompt_tokens"] if metrics else 0,
            completion_tokens=metrics["completion_tokens"] if metrics else 0,
            total_tokens=metrics["total_tokens"] if metrics else 0,
            cost=metrics["cost"] if metrics else 0.0,
            avg_similarity_score=avg_score,
            vector_store_size=db_stats.get("total_chunks", 0),
            total_documents_indexed=db_stats.get("unique_files_count", 0),
            scores=scores,
            original_query=prompt,
            rewritten_query=retrieval_res.get("rewritten_query", retrieval_res["query"]),
            similarity_improvement=retrieval_res.get("similarity_improvement", 0.0),
            strategy=retrieval_strategy
        )
        
        # Keep session state history in sync
        st.session_state.metrics_history.append(log_entry)

        st.rerun()

# ----------------- TAB 2: SOURCE ATTRIBUTION -----------------

with tab_sources:
    st.markdown("### Document Sources & Citations")
    res = st.session_state.retrieved_results
    
    if not res:
        st.info("Ask a question to see the matching search chunks and similarity scores.")
    else:
        original_q = res.get("original_query", res["query"])
        rewritten_q = res.get("rewritten_query", res["query"])
        
        col_q1, col_q2 = st.columns(2)
        col_q1.markdown(f"**Original Query:** `{original_q}`")
        col_q2.markdown(f"**Rewritten Query (Optimized):** `{rewritten_q}`")
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        col_meta1.metric("Retrieval Latency", f"{res['latency']:.4f} seconds")
        col_meta2.metric("Source Chunks Found", f"{len(res['original_chunks'])}")
        col_meta3.metric("Chunks post-Compression", f"{len(res['chunks'])}")
        
        st.markdown("---")
        
        # Display each matching document
        for idx, doc in enumerate(res["original_chunks"]):
            score = res["scores"][idx] if idx < len(res["scores"]) else 0.0
            
            # Check if this document has a compressed version
            comp_content = None
            if use_compressor and idx < len(res["chunks"]):
                comp_content = res["chunks"][idx].page_content
                
            filename = doc.metadata.get("filename", "Unknown File")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            section = doc.metadata.get("section_title", "Document Context")
            source_path = doc.metadata.get("source", "N/A")
            
            with st.expander(f"📄 [{idx+1}] {filename} — Section: {section} (Similarity: {score:.4f})"):
                st.markdown(f"**Metadata:**")
                st.write(f"- **Chunk ID:** `{chunk_id}`")
                st.write(f"- **Source File Path:** `{source_path}`")
                st.write(f"- **Index Timestamp:** `{doc.metadata.get('timestamp')}`")
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**Original Chunk Text:**")
                    st.code(doc.page_content, language="markdown")
                with col_t2:
                    st.markdown("**Compressed Chunk Text (Fed to LLM):**")
                    if comp_content:
                        st.code(comp_content, language="markdown")
                    else:
                        st.info("Compression was disabled. Full chunk text was used.")

# ----------------- TAB 3: DATABASE HEALTH -----------------

with tab_health:
    st.markdown("### Vector Database Catalog Status")
    
    stats = st.session_state.chroma_manager.get_collection_stats()
    
    # Render indicators using CSS cards
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    
    status = stats["status"]
    status_class = "status-healthy" if status == "Healthy" else ("status-empty" if status == "Empty" else "status-error")
    
    with col_h1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Database Status</div>"
            f"<div class='metric-value'><span class='status-badge {status_class}'>{status}</span></div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_h2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Indexed Files</div>"
            f"<div class='metric-value'>{stats['unique_files_count']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_h3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Total Text Chunks</div>"
            f"<div class='metric-value'>{stats['total_chunks']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_h4:
        # Calculate approximate file size of documents folder
        total_bytes = 0
        for root, _, files in os.walk("docs"):
            for file in files:
                total_bytes += os.path.getsize(os.path.join(root, file))
        formatted_fsize = format_size(total_bytes)
        
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Raw Docs Size</div>"
            f"<div class='metric-value'>{formatted_fsize}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    st.markdown("### Catalog Inventory Details")
    
    if not stats["indexed_files"]:
        st.info("The vector database is currently empty. Upload documents (.md, .pdf) or rebuild the index to populate files.")
    else:
        # Build Pandas Table showing file inventories
        rows = []
        for path, info in stats["indexed_files"].items():
            rows.append({
                "Filename": info["filename"],
                "File Hash": info["file_hash"][:12] + "...",
                "Index Timestamp": info["timestamp"],
                "Total Chunks": info["chunk_count"],
                "Disk Path": path
            })
            
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

# ----------------- TAB 4: RETRIEVAL ANALYTICS DASHBOARD -----------------

with tab_analytics:
    st.markdown("### 📊 Retrieval Analytics Dashboard")
    
    history = st.session_state.metrics_history
    
    col_dash1, col_dash2 = st.columns([8, 2])
    with col_dash2:
        if st.button("🗑️ Clear Analytics Log", use_container_width=True):
            st.session_state.metrics_tracker.clear_history()
            st.session_state.metrics_history = []
            st.rerun()
            
    if not history:
        st.info("Run search queries in the chat assistant to display latency and similarity analytics.")
    else:
        df_metrics = pd.DataFrame(history)
        
        # Ensure all expected columns exist to support older schema versions from disk
        expected_cols = {
            "retrieval_latency": 0.0,
            "generation_latency": 0.0,
            "total_latency": 0.0,
            "avg_similarity_score": 0.0,
            "similarity_improvement": 0.0,
            "retrieved_chunks": 0,
            "compressed_chunks": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "query": "",
            "strategy": "Similarity Search"
        }
        for col, default in expected_cols.items():
            if col not in df_metrics.columns:
                df_metrics[col] = default
                
        df_metrics["Time"] = pd.to_datetime(df_metrics["timestamp"], unit="s").dt.strftime('%H:%M:%S')
        
        # Display Totals Card Metrics
        st.markdown("##### 📈 Key Performance Indicators (KPIs)")
        
        # Row 1 of KPIs
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        avg_retrieval = df_metrics["retrieval_latency"].mean() if "retrieval_latency" in df_metrics.columns else 0.0
        avg_generation = df_metrics["generation_latency"].mean() if "generation_latency" in df_metrics.columns else 0.0
        avg_total = df_metrics["total_latency"].mean() if "total_latency" in df_metrics.columns else 0.0
        avg_sim = df_metrics["avg_similarity_score"].mean() if "avg_similarity_score" in df_metrics.columns else 0.0
        
        col_m1.metric("Avg Retrieval Time", f"{avg_retrieval:.4f}s")
        col_m2.metric("Avg Generation Time", f"{avg_generation:.4f}s")
        col_m3.metric("Avg Total Response", f"{avg_total:.4f}s")
        col_m4.metric("Avg Similarity Score", f"{avg_sim:.4f}")
        
        st.write("") # Spacer
        
        # Row 2 of KPIs
        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        queries_count = len(df_metrics)
        avg_chunks = df_metrics["retrieved_chunks"].mean() if "retrieved_chunks" in df_metrics.columns else 0.0
        
        # Fetch db stats for real-time vector store sizes
        db_stats = st.session_state.chroma_manager.get_collection_stats()
        db_chunks = db_stats.get("total_chunks", 0)
        db_docs = db_stats.get("unique_files_count", 0)
        
        col_m5.metric("Queries Executed", f"{queries_count}")
        col_m6.metric("Avg Chunks Retrieved", f"{avg_chunks:.1f}")
        col_m7.metric("Vector Store Size", f"{db_chunks} chunks")
        col_m8.metric("Docs Indexed", f"{db_docs}")
        
        st.markdown("---")
        
        # Strategy Comparison Metrics section
        st.markdown("##### ⚖️ Strategy Comparison Metrics")
        if "strategy" in df_metrics.columns:
            comparison_df = df_metrics.groupby("strategy").agg({
                "query": "count",
                "retrieval_latency": "mean",
                "avg_similarity_score": "mean",
                "similarity_improvement": "mean"
            }).reset_index()
            
            comparison_df.rename(columns={
                "strategy": "Retrieval Strategy",
                "query": "Queries Executed",
                "retrieval_latency": "Avg Retrieval Time (s)",
                "avg_similarity_score": "Avg Similarity Score",
                "similarity_improvement": "Avg Similarity Improvement"
            }, inplace=True)
            
            # Format numeric columns nicely
            comparison_df["Avg Retrieval Time (s)"] = comparison_df["Avg Retrieval Time (s)"].map(lambda x: f"{x:.4f}s")
            comparison_df["Avg Similarity Score"] = comparison_df["Avg Similarity Score"].map(lambda x: f"{x:.4f}")
            comparison_df["Avg Similarity Improvement"] = comparison_df["Avg Similarity Improvement"].map(lambda x: f"{x:+.4f}")
            
            st.dataframe(comparison_df.set_index("Retrieval Strategy"), use_container_width=True)
        else:
            st.info("No strategy comparison data available yet.")
            
        st.markdown("---")
        
        st.markdown("##### 📊 Performance Visualizations")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("**Response Time History (Seconds)**")
            chart_df = df_metrics[["Time"]].copy()
            chart_df["Retrieval Time"] = df_metrics["retrieval_latency"] if "retrieval_latency" in df_metrics.columns else df_metrics["total_latency"] * 0.3
            chart_df["Generation Time"] = df_metrics["generation_latency"] if "generation_latency" in df_metrics.columns else df_metrics["total_latency"] * 0.7
            chart_df["Total Time"] = df_metrics["total_latency"]
            st.line_chart(chart_df.set_index("Time"), height=300)
            
        with col_chart2:
            st.markdown("**Retrieval Score Distribution & Improvement**")
            # Create a chart showing similarity scores and improvements over queries
            score_chart_df = df_metrics[["Time"]].copy()
            score_chart_df["Similarity Score"] = df_metrics["avg_similarity_score"] if "avg_similarity_score" in df_metrics.columns else 0.0
            if "similarity_improvement" in df_metrics.columns:
                score_chart_df["Improvement"] = df_metrics["similarity_improvement"]
            else:
                score_chart_df["Improvement"] = 0.0
            st.bar_chart(score_chart_df.set_index("Time"), height=300)
            
        st.markdown("---")
        
        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            st.markdown("**Token Consumption Over Time**")
            chart_tokens = df_metrics[["Time", "prompt_tokens", "completion_tokens"]].copy()
            chart_tokens.rename(columns={
                "prompt_tokens": "Input Tokens (Prompt)",
                "completion_tokens": "Output Tokens (Completion)"
            }, inplace=True)
            st.bar_chart(chart_tokens.set_index("Time"), height=300)
            
        with col_chart4:
            st.markdown("**Total Cost Analysis ($)**")
            cost_chart = df_metrics[["Time", "cost"]].copy()
            cost_chart.rename(columns={"cost": "Estimated Cost ($)"}, inplace=True)
            st.line_chart(cost_chart.set_index("Time"), height=300)

        st.markdown("---")
        st.markdown("### Historical Query Logs")
        
        # Format logs displaying all new and old metrics
        log_df = df_metrics[["Time", "query", "retrieved_chunks", "total_latency", "avg_similarity_score", "similarity_improvement"]].copy()
        log_df.rename(columns={
            "query": "User Question",
            "retrieved_chunks": "Chunks Retrieved",
            "total_latency": "Latency (s)",
            "avg_similarity_score": "Avg Similarity",
            "similarity_improvement": "Rewriter Improvement"
        }, inplace=True)
        st.dataframe(log_df.set_index("Time"), use_container_width=True)
