# System Architecture

DocuMind AI utilizes a modern, modular Retrieval-Augmented Generation (RAG) architecture.

## Component Block Diagram

The application consists of the following key sub-systems:

1. **Ingestion Loader:** Recursively scans the `docs/` folder, hashes content using SHA-256 for duplicate checks, and compares files against an index manifest to support incremental loading.
2. **Text Splitter:** Uses LangChain's `RecursiveCharacterTextSplitter` to parse Markdown files into overlapping chunks of 1000 characters by default. It annotates each chunk with source paths and section titles.
3. **Embedding Model:** Interacts with OpenAI's `text-embedding-3-small` API to convert text chunks into high-dimensional numerical vectors.
4. **Vector Store:** ChromaDB persists document vectors and metadata inside the local `chroma_db/` folder.
5. **Retriever Layer:** Offers Cosine Similarity, Maximum Marginal Relevance (MMR), and Score Threshold search strategies. Contains a Query Rewriter to resolve conversation context and a Context Compressor to trim non-relevant text.
6. **Generator Service:** Calls GPT-4o-mini to synthesize clear, inline-cited responses using only the retrieved documents.

## Database Location

All SQLite data and vector databases are persisted locally under the project's root folder:
```
chroma_db/
```
You can delete or rebuild the database index at any time from the Streamlit sidebar.
