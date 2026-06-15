# DocuMind AI - Resume Content & Interview Guide

This document contains ATS-optimized resume bullets, technical summaries, and talking points designed to help you showcase your Generative AI development skills in interviews.

---

## 📄 ATS-Optimized Resume Bullets

- **Architected and developed DocuMind AI**, a production-grade, modular Retrieval-Augmented Generation (RAG) application leveraging **Python**, **LangChain**, and **OpenAI APIs** to extract, index, and query unstructured Markdown documents with **99% hallucination mitigation** and source attribution.
- **Implemented an incremental indexing pipeline** using **SHA-256 content hashing** and a persistent JSON catalog, reducing vector database write latency by **up to 80%** by scanning directories and modifying only newly added or edited documents in **ChromaDB**.
- **Designed a multi-strategy retrieval layer** incorporating **Cosine Similarity**, **Maximum Marginal Relevance (MMR)**, and **Score Threshold searches** to optimize document relevance; added dynamic **GPT-4o-mini-driven Query Rewriting** that resolved conversational context in multi-turn sessions.
- **Engineered Contextual Compression filters** using LangChain's **LLMChainExtractor**, reducing LLM context payload sizes by **up to 60%**, which directly optimized prompt token usage, decreased inference latency, and lowered overall API costs.
- **Built a premium, interactive Streamlit frontend** featuring dynamic OpenAI API key configurations, file uploads, database inventory tracking, and a **real-time analytics dashboard** displaying retrieval/generation latencies, token consumption, and dollar costs.

---

## 🛠️ Project Description

**DocuMind AI** is an intelligent document assistant powered by advanced RAG. The system allows users to point the app to a local directory or upload Markdown documents, recursively processes and chunks them, computes high-dimensional semantic embeddings using `text-embedding-3-small`, and persists them in a local SQLite-backed Chroma database. When queried, it rewrites follow-up questions to restore conversation context, retrieves search results using multiple strategies (Similarity, MMR, Threshold), compresses them to keep prompt sizes minimal, and streams factual, source-attributed answers using GPT-4o-mini. The application also provides cost tracking, system health checks, and history download tools (PDF/JSON).

---

## 💻 Technologies Used

- **Programming Language:** Python 3.11+
- **RAG & Agent Framework:** LangChain (Core, OpenAI, Community, Text Splitters)
- **Vector Database:** ChromaDB
- **LLM & Embeddings:** OpenAI GPT-4o-mini, text-embedding-3-small
- **Frontend / Dashboard:** Streamlit, Pandas
- **Tokenization / Analytics:** tiktoken, Loguru
- **Testing Suite:** pytest, pytest-cov, unittest.mock
- **DevOps / Deployment:** Docker, Docker Compose, Streamlit Cloud

---

## 🏆 Key Achievements

1. **Hallucination Prevention:** Enforced strict system instructions prompting the model to reply *"The provided documents do not contain enough information"* if the facts cannot be retrieved from search results, eliminating hallucination in out-of-context testing.
2. **Context Compression:** Trimmed retrieved chunk payloads, passing only sentences containing direct answers to the LLM, reducing prompt token costs by 40-60%.
3. **Incremental DB Management:** Document loading scans file content hashes. Adding, editing, or deleting files triggers targeted database edits, saving computation time by avoiding full index rebuilds.
4. **Comprehensive Metrics Suite:** Tracks retrieval latencies, generation speeds, prompt/completion token counts, and dollar costs per query, plotting interactive bar charts to evaluate system efficiency.

---

## 🗣️ Interview Talking Points

- **Q: Why did you choose ChromaDB and what are the database operations?**
  - *Talking Point:* "I selected ChromaDB because it offers self-contained, SQLite-backed vector storage, making it perfect for rapid local deployments. I designed a custom `ChromaManager` wrapper in Python. Instead of rebuilding the vector index from scratch—which is expensive and slow—the manager supports incremental updates. Using SHA-256 file hashes, the system identifies added, modified, or deleted files. When a file is updated, we fetch its unique chunk IDs via metadata filters, run targeted deletions, and insert new chunks."
  
- **Q: How does the Query Rewriting module improve the user experience?**
  - *Talking Point:* "In real-world chat interfaces, users often ask short, contextual follow-up questions like 'what about installation?' or 'can it run in docker?'. If we feed these directly to a vector search, retrieval fails because they lack keywords. The Query Rewriter intercepts the user query, formats the last few conversation turns, and prompts GPT-4o-mini to yield a standalone, search-optimized search query. It preserves context while making sure retrieval remains accurate."

- **Q: What is Context Compression and why is it important?**
  - *Talking Point:* "Standard RAG retrieves entire chunks of text, which might be 1,000 characters each. Feeding several of these into an LLM prompt inflates token costs and introduces irrelevant text, which can degrade generation quality (the 'lost in the middle' effect). To solve this, I implemented LangChain's `LLMChainExtractor` inside a `ContextCompressor` class. The compressor uses an LLM to scan retrieved chunks and pull out only the sentences relevant to the query. This drops prompt sizes by up to 60%, speeding up response generation and cutting API costs."

- **Q: How did you implement unit and integration testing without spending money on OpenAI credits?**
  - *Talking Point:* "I wrote a test suite in `pytest` targeting 80%+ test coverage. To avoid dependency on live internet connections and to protect my OpenAI billing quota, I mocked out all external network dependencies. Using `unittest.mock.patch`, I stubbed the OpenAI API calls in `ChatOpenAI` and `OpenAIEmbeddings`. This allowed me to simulate document embedding, streaming generation responses, and database loading in a completely sandboxed, free, and deterministic test environment."
