# DocuMind AI 🧠

### *Intelligent Knowledge Assistant Powered by Production-Grade RAG*

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://github.com/chroma-core/chroma)
[![Ollama](https://img.shields.io/badge/Ollama-LocalLLM-gray.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DocuMind AI is a state-of-the-art, enterprise-grade **Retrieval-Augmented Generation (RAG)** application built to ingest, segment, index, and query document files (Markdown, PDF) completely locally using local models (via Ollama) or globally using OpenAI services. 

DocuMind AI features a beautiful, custom-styled Streamlit interface, an **OOP Strategy-based Retrieval Engine**, a **Standalone Query Rewriter**, detailed **Sources & Attribution Confidence Badges**, a comprehensive **Retrieval Analytics Dashboard** showing strategy comparisons, and multi-format **Conversation Exporters** (PDF, TXT, MD, JSON).

---

## 🏗️ Architecture Flow

Below is the conceptual execution flowchart of the system, demonstrating how user questions are parsed, rewritten, retrieved polymorphically, compressed, answered, logged, and compared:

```mermaid
graph TD
    User([User Query]) --> RewriterToggle{Enable Query Rewriting?}
    RewriterToggle -- Yes --> Rewriter["Query Rewriter (OpenAI / Ollama)"]
    RewriterToggle -- No --> QueryPass[Original Query]
    Rewriter -- Standalone Search Query --> Retriever[DocuMind Retriever]
    QueryPass --> Retriever
    
    subgraph Strategy Pattern (OOP)
        Retriever --> StrategySelector{Selected Strategy}
        StrategySelector --> Sim[SimilaritySearchStrategy]
        StrategySelector --> MMR[MMRSearchStrategy]
        StrategySelector --> Thresh[ScoreThresholdSearchStrategy]
    end

    subgraph Vector DB
        Sim --> ChromaDB[(Chroma Vector DB)]
        MMR --> ChromaDB
        Thresh --> ChromaDB
    end

    ChromaDB --> Filter[Retrieved Document Chunks]
    Filter --> CompressorToggle{Context Compression?}
    CompressorToggle -- Yes --> Compressor["Context Compressor (LLMChainExtractor)"]
    CompressorToggle -- No --> Prompt[Prompt Builder]
    Compressor --> Prompt
    
    Prompt --> Generator["LLM Generator (OpenAI / Ollama)"]
    Generator --> Response[Streaming Response]
    Response --> UI[Streamlit UI]
    
    subgraph Analytics & Export Layer
        UI --> Metrics[Metrics Tracker]
        Metrics --> LogFile[(metrics_history.json)]
        Metrics --> Dashboard[Retrieval Analytics Dashboard]
        UI --> Exporters["Exporters (PDF, MD, TXT, JSON)"]
    end
```

---

## 🌟 Features

### 🚀 1. Production-Grade Ingestion & Splitting
- **Multi-Format Processing:** Parses structured Markdown (`.md`) and rich PDF (`.pdf`) documents page-by-page.
- **Section-Aware Splitter:** Automatically associates each chunk with its closest parent section heading (`# Heading`, `## Subheading`) to preserve local context.
- **Duplicate & Modification Manifest:** Tracks SHA-256 hashes inside `chroma_db/index_manifest.json` to prevent duplicate index processing, handling document additions, modifications, and deletions recursively.
- **Adjustable Splitting parameters:** Real-time controls for chunk size and overlapping parameters from the UI sidebar.

### 🔌 2. Dual-Provider Model Flexibility (OpenAI & Ollama)
- **OpenAI Integration:** Powered by `gpt-4o-mini` and `text-embedding-3-small` for fast, cost-efficient cloud execution.
- **100% Local Deployment:** Powered by Ollama (`llama3.2:3b` for LLM, `nomic-embed-text` for embeddings) for complete data privacy and offline processing at zero dollar cost.
- **Hot-Reloading configuration:** Switch providers, models, and API keys dynamically without restarting the application.

### 🎯 3. OOP Strategy-based Retrieval Engine
- **Strategy Pattern Architecture:** Encapsulates vector searches into polymorphic classes (`SimilaritySearchStrategy`, `MMRSearchStrategy`, and `ScoreThresholdSearchStrategy`).
- **MMR Search:** Max Marginal Relevance balances match similarity and document diversity to reduce redundancy.
- **Score Threshold Search:** Drops irrelevant chunks scoring below a user-defined threshold.
- **Inline Confidence Badging:** Labels source chunks using strict confidence limits:
  - 🟢 **High Confidence** ($>0.85$ Similarity)
  - 🟡 **Medium Confidence** ($0.65$ to $0.85$ Similarity)
  - 🔴 **Low Confidence** ($<0.65$ Similarity)

### 🔄 4. Query Rewriter & Optimization
- **Typo & Vague Query Resolution:** Automatically rewrites brief or vague queries into standalone search questions (e.g., `"what about education"` expands into `"What educational qualifications are present in the uploaded documents?"`).
- **Similarity Improvement Analytics:** Performs a background query search using the raw question to compute and visualize semantic score improvements (e.g., `+12.4%`).
- **Attribution Transparency:** Displays a comparison banner showing the original raw input side-by-side with the optimized standalone query.

### 📊 5. Retrieval Analytics Dashboard
- **Interactive Metric Cards:** Custom glassmorphic Streamlit metric cards mapping Retrieval Time, Generation Time, Total Time, Avg Similarity Score, Chunks Retrieved, DB sizes, and Indexed Document counts.
- **Strategy Comparison Metrics:** An interactive grouped table computing average latency and similarity scores for each strategy used in the session.
- **Historical Performance Graphs:** Line charts of multi-metric latency histories, bar charts of chunk score distributions, token counts, and API costs.

### 💾 6. Multi-Format Exporters
- **TXT Export:** Exports plain-text conversation transcripts including user questions, assistant answers, and source previews.
- **Markdown Export:** Generates rich markdown transcripts quoting sources inside code segments and blockquotes.
- **PDF Export:** Creates a structured, styled PDF document utilizing color-coded role tags (indigo for user, teal for assistant) and italicized citations.

---

## 📂 Folder Structure

```
documind-ai/
├── app.py                      # Main Streamlit UI
├── requirements.txt            # Package dependencies
├── README.md                   # Project Documentation
├── .env.example                # Environmental variables template
├── Dockerfile                  # Application build script
├── docker-compose.yml          # Container configuration
│
├── docs/                       # Directory for ingested Markdown and PDF documents
├── chroma_db/                  # Persistent database storage & index manifests
├── logs/                       # Application logs (app.log, metrics_history.json)
│
├── src/                        # Source Code
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py           # Multi-format scanners (PDF/MD)
│   │   └── parser.py           # Normalization & preprocessing
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── splitter.py         # Heading-aware recursive splitting
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py # OpenAI & Ollama embedding layer
│   ├── vectordb/
│   │   ├── __init__.py
│   │   └── chroma_manager.py   # Collection CRUD controls
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── strategies.py       # OOP Retrieval strategies (Similarity, MMR, Thresh)
│   │   ├── query_rewriter.py   # Standing context query rewriter
│   │   ├── compressor.py       # Context-compression extractor
│   │   └── retriever.py        # Retrieval strategy pattern coordinator
│   ├── llm/
│   │   ├── __init__.py
│   │   └── generator.py        # Generative model response streamer
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Latency, token, and cost logger
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Loguru rotation settings
│       ├── exporters.py        # TXT, Markdown, and PDF exporter utility
│       └── helpers.py          # Math calculations & size formatting
│
└── tests/                      # Pytest Suite
    ├── __init__.py
    ├── test_ingestion.py       # Loader and parser checks
    ├── test_embeddings.py      # Embedding batching tests
    ├── test_retrieval.py       # Search strategies tests
    └── test_pipeline.py        # End-to-end integration tests (Ollama/OpenAI)
```

---

## 📸 Screenshots Section

Below is a representation of the premium dashboard interface.

```
+---------------------------------------------------------------------------------------------------+
| 🧠 DocuMind AI  [💬 Chat Assistant]  [📑 Source Attribution]  [🏥 DB Health]  [📈 Analytics Tab]   |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  🎯 Strategy Selected: Similarity Search                                                          |
|                                                                                                   |
|  +--------------------+  +--------------------+  +--------------------+  +--------------------+   |
|  |   RETRIEVAL TIME   |  |  GENERATION TIME   |  | TOTAL RESPONSE TIME|  |AVG SIMILARITY SCORE|   |
|  |     0.0345s        |  |     1.2405s        |  |     1.2750s        |  |     0.8845         |   |
|  +--------------------+  +--------------------+  +--------------------+  +--------------------+   |
|  |   CHUNKS RETRIEVED |  | VECTOR STORE SIZE  |  |    DOCS INDEXED    |  |RETRIEVAL IMPOVEMT  |   |
|  |          4         |  |    32 chunks       |  |      3 files       |  |     +12.45%        |   |
|  +--------------------+  +--------------------+  +--------------------+  +--------------------+   |
|                                                                                                   |
|  ⚖️ Strategy Comparison Metrics                                                                    |
|  +---------------------------+------------------+---------------------+-----------------------+   |
|  | Retrieval Strategy        | Queries Executed | Avg Retrieval Time  | Avg Similarity Score  |   |
|  +---------------------------+------------------+---------------------+-----------------------+   |
|  | Similarity Search         |        4         |       0.0321s       |        0.8710         |   |
|  | MMR Search                |        2         |       0.0450s       |        0.8415         |   |
|  +---------------------------+------------------+---------------------+-----------------------+   |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 🛠️ Installation & Setup

### Local Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/username/documind-ai.git
   cd documind-ai
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```

3. **Install Core Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and fill in your OpenAI key:
   ```env
   OPENAI_API_KEY=sk-yourOpenAiKeyHere...
   ```

5. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

---

## 🐳 Docker Container Setup

Run the entire application in a docker container. Docker Compose binds local directories, ensuring your database (`chroma_db/`), indexed documents (`docs/`), and application logs (`logs/`) persist.

1. **Verify your `.env` contains your OpenAI key.**
2. **Build and Start Container:**
   ```bash
   docker-compose up --build
   ```
3. **Open browser:** Navigate to `http://localhost:8501`.
4. **Shutdown Containers:**
   ```bash
   docker-compose down
   ```

---

## ⚙️ Ollama Local Model Setup

To run DocuMind AI completely locally with zero cost:

1. **Download & Install Ollama:** Follow the setup guide on [Ollama's official website](https://ollama.com/).
2. **Download Models:** Open a terminal and run:
   ```bash
   # Pull the local chat LLM
   ollama pull llama3.2:3b
   
   # Pull the local embedding model
   ollama pull nomic-embed-text
   ```
3. **Verify the server is running:** Navigate to `http://localhost:11434` in your browser. It should say *"Ollama is running"*.
4. **Switch configuration in the sidebar:**
   - LLM Provider: `Ollama (Local)`
   - Ollama Model Name: `llama3.2:3b`

---

## 🗄️ ChromaDB Configuration

- **SQLite Version:** ChromaDB requires SQLite $\ge 3.35.0$. 
  - **Linux/macOS fix:** If you encounter database initialization issues, run `pip install pysqlite3-binary` and add the following at the very top of `app.py`:
    ```python
    import sys
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    ```
- **Database Persistence:** Vectors, collections, and catalog manifest files are saved under the `chroma_db/` directory, which can be safely cleared using the **🗑️ Delete Index** button in the sidebar.

---

## 📄 Resume Highlights (DocuMind Indexing Demo)

DocuMind AI is pre-configured to parse and index PDF resumes for evaluation:
- **Index Target:** `docs/Nishita_Rana_Resume_ (1).pdf`
- **Vague queries optimization:**
  - *"What about education?"* is rewritten to ask for educational qualifications.
  - *"What is experience?"* is rewritten to ask for professional experience.
- **Source attribution:** Displays confidence metrics for segments extracted from the resume PDF.

---

## 🔮 Future Improvements

1. **Hybrid Keyword + Vector Search (BM25 + Semantic):** Fuse vector similarity search results with sparse BM25 keyword matching for optimal search relevance.
2. **Reranking Layer (Cohere Rerank / Cross-Encoder):** Pass retrieved chunks through a reranking transformer model to optimize top-k relevancy before generating answers.
3. **Fine-grained User Access Controls:** Add secure login/authentication interfaces for enterprise deployments.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
