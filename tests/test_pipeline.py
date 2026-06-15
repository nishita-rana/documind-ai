import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.ingestion.parser import MarkdownParser
from src.chunking.splitter import MarkdownSplitter
from src.vectordb.chroma_manager import ChromaManager
from src.retrieval.retriever import DocuMindRetriever
from src.llm.generator import LLMGenerator
from src.evaluation.metrics import MetricsTracker

@patch("src.embeddings.embedding_service.OpenAIEmbeddings")
@patch("src.vectordb.chroma_manager.Chroma")
@patch("src.llm.generator.ChatOpenAI")
def test_full_pipeline(mock_chat_cls, mock_chroma_cls, mock_embeddings_cls, tmp_path):
    # Mock LLM and Embeddings
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    
    mock_embeddings = MagicMock()
    mock_embeddings_cls.return_value = mock_embeddings
    
    # Mock Chroma vector store
    mock_store = MagicMock()
    mock_chroma_cls.return_value = mock_store

    # 1. Document Parsing & Chunking
    parser = MarkdownParser()
    doc_dict = {
        "content": "# Setup Guide\n\nThis is step one to install DocuMind AI.\n\n## Verification\nMake sure it runs.",
        "path": "/docs/setup.md",
        "filename": "setup.md",
        "file_hash": "abc123hash",
        "timestamp": "2026-06-15T12:00:00"
    }
    
    splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=50)
    chunks, stats = splitter.split_document(doc_dict)
    
    assert len(chunks) == 2
    assert stats["total_chunks"] == 2
    assert chunks[0].metadata["section_title"] == "Setup Guide"
    assert chunks[1].metadata["section_title"] == "Verification"

    # 2. Chroma Manager Integration
    from src.embeddings.embedding_service import OpenAIEmbeddingService
    embed_service = OpenAIEmbeddingService(api_key="mock_key")
    chroma_manager = ChromaManager(persist_dir=str(tmp_path / "db"), collection_name="test_col", embedding_service=embed_service)
    
    chroma_manager.add_documents(chunks)
    mock_store.add_documents.assert_called_once()

    # 3. Retrieval Integration
    # Setup retrieve mock return values
    mock_store.similarity_search_with_score.return_value = [
        (chunks[0], 0.3)
    ]
    
    retriever = DocuMindRetriever(chroma_manager=chroma_manager, api_key="mock_key", use_rewriter=False, use_compressor=False)
    retrieval_res = retriever.retrieve("How to install?", strategy="Similarity Search", k=1)
    
    assert retrieval_res["count"] == 1
    assert retrieval_res["chunks"][0].metadata["chunk_id"] == "abc123hash_c0"
    assert retrieval_res["chunks"][0].page_content == "This is step one to install DocuMind AI."

    # 4. LLM Response Generation Mocking
    # Mock generation stream yields
    mock_stream_chunk1 = MagicMock()
    mock_stream_chunk1.content = "To install DocuMind AI, follow "
    mock_stream_chunk2 = MagicMock()
    mock_stream_chunk2.content = "step one of the setup guide."
    
    mock_llm.stream.return_value = [mock_stream_chunk1, mock_stream_chunk2]

    generator = LLMGenerator(api_key="mock_key")
    response_stream = generator.generate_response(
        query=retrieval_res["query"],
        context_documents=retrieval_res["chunks"],
        chat_history=[]
    )
    
    streamed_tokens = []
    metrics = None
    for chunk in response_stream:
        if chunk["type"] == "text":
            streamed_tokens.append(chunk["content"])
        elif chunk["type"] == "metrics":
            metrics = chunk

    response_text = "".join(streamed_tokens)
    assert response_text == "To install DocuMind AI, follow step one of the setup guide."
    assert metrics is not None
    assert metrics["prompt_tokens"] > 0
    assert metrics["completion_tokens"] > 0
    assert metrics["cost"] > 0.0

    # 5. Metrics Tracking
    history_file = tmp_path / "metrics.json"
    metrics_tracker = MetricsTracker(history_file=str(history_file))
    
    logged_run = metrics_tracker.log_run(
        query="How to install?",
        retrieved_count=len(retrieval_res["original_chunks"]),
        compressed_count=len(retrieval_res["chunks"]),
        retrieval_latency=retrieval_res["latency"],
        generation_latency=0.15,
        prompt_tokens=metrics["prompt_tokens"],
        completion_tokens=metrics["completion_tokens"],
        total_tokens=metrics["total_tokens"],
        cost=metrics["cost"]
    )
    
    assert logged_run["retrieved_chunks"] == 1
    assert logged_run["compressed_chunks"] == 1
    assert logged_run["total_tokens"] == metrics["prompt_tokens"] + metrics["completion_tokens"]
    
    # Reload from disk history and verify
    history = metrics_tracker.load_history()
    assert len(history) == 1
    assert history[0]["query"] == "How to install?"


@patch("src.embeddings.embedding_service.OllamaEmbeddings", create=True)
@patch("src.vectordb.chroma_manager.Chroma")
@patch("src.llm.generator.ChatOllama", create=True)
def test_ollama_pipeline(mock_chat_cls, mock_chroma_cls, mock_embeddings_cls, tmp_path):
    # Mock LLM and Embeddings
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    
    mock_embeddings = MagicMock()
    mock_embeddings_cls.return_value = mock_embeddings
    
    # Mock Chroma vector store
    mock_store = MagicMock()
    mock_chroma_cls.return_value = mock_store

    from src.embeddings.embedding_service import OpenAIEmbeddingService
    embed_service = OpenAIEmbeddingService(provider="ollama", model="llama3.1")
    chroma_manager = ChromaManager(persist_dir=str(tmp_path / "db"), collection_name="test_col_ollama", embedding_service=embed_service)
    
    # Retrieval Integration
    mock_store.similarity_search_with_score.return_value = [
        (Document(page_content="Ollama runs locally.", metadata={"chunk_id": "c0"}), 0.5)
    ]
    
    retriever = DocuMindRetriever(chroma_manager=chroma_manager, provider="ollama", model="llama3.1", use_rewriter=False, use_compressor=False)
    retrieval_res = retriever.retrieve("Ollama?", strategy="Similarity Search", k=1)
    
    assert retrieval_res["count"] == 1
    assert retrieval_res["chunks"][0].page_content == "Ollama runs locally."

    # LLM Response Generation Mocking
    mock_stream_chunk1 = MagicMock()
    mock_stream_chunk1.content = "Ollama answer."
    mock_llm.stream.return_value = [mock_stream_chunk1]

    generator = LLMGenerator(provider="ollama", model="llama3.1")
    response_stream = generator.generate_response(
        query=retrieval_res["query"],
        context_documents=retrieval_res["chunks"],
        chat_history=[]
    )
    
    streamed_tokens = []
    metrics = None
    for chunk in response_stream:
        if chunk["type"] == "text":
            streamed_tokens.append(chunk["content"])
        elif chunk["type"] == "metrics":
            metrics = chunk

    response_text = "".join(streamed_tokens)
    assert response_text == "Ollama answer."
    assert metrics is not None
    assert metrics["cost"] == 0.0  # Local model has zero cost

