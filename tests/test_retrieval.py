import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.compressor import ContextCompressor
from src.retrieval.retriever import DocuMindRetriever

@patch("src.retrieval.query_rewriter.ChatOpenAI")
def test_query_rewriter(mock_chat_cls):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "How do I install the software?"
    mock_llm.invoke.return_value = mock_response
    mock_chat_cls.return_value = mock_llm

    rewriter = QueryRewriter(api_key="mock_key")
    
    # 1. No history should skip LLM call
    query = "installation details"
    rewritten1 = rewriter.rewrite_query(query, [])
    assert rewritten1 == query
    mock_llm.invoke.assert_not_called()

    # 2. History present should invoke LLM
    history = [
        {"role": "user", "content": "I want to install DocuMind AI"},
        {"role": "assistant", "content": "Sure, I can help with that."}
    ]
    rewritten2 = rewriter.rewrite_query("what about Windows?", history)
    assert rewritten2 == "How do I install the software?"
    mock_llm.invoke.assert_called_once()

@patch("src.retrieval.compressor.ChatOpenAI")
@patch("src.retrieval.compressor.LLMChainExtractor")
def test_context_compressor(mock_extractor_cls, mock_chat_cls):
    mock_extractor = MagicMock()
    doc = Document(page_content="Only this sentence is relevant. The rest is fluff.", metadata={"chunk_id": "c1"})
    mock_extractor.compress_documents.return_value = [
        Document(page_content="Only this sentence is relevant.", metadata={"chunk_id": "c1"})
    ]
    mock_extractor_cls.from_llm.return_value = mock_extractor

    compressor = ContextCompressor(api_key="mock_key")
    compressed = compressor.compress_documents([doc], "Only relevant sentence")
    
    assert len(compressed) == 1
    assert compressed[0].page_content == "Only this sentence is relevant."
    assert compressed[0].metadata["chunk_id"] == "c1"

def test_retriever_strategies():
    # Setup mock ChromaManager & VectorStore
    mock_chroma = MagicMock()
    mock_store = MagicMock()
    mock_chroma.vector_store = mock_store
    
    # Mock documents returned by search
    doc1 = Document(page_content="Text chunk 1", metadata={"chunk_id": "id1", "source": "file.md", "file_hash": "h1"})
    doc2 = Document(page_content="Text chunk 2", metadata={"chunk_id": "id2", "source": "file.md", "file_hash": "h1"})
    
    # Similarity Search return: list of tuples (Document, L2 distance)
    # L2 distance = 0.5 -> mapped similarity score = 1 / (1 + 0.5) = 0.6667
    mock_store.similarity_search_with_score.return_value = [(doc1, 0.5), (doc2, 1.5)]
    
    # MMR search returns list of Documents
    mock_store.max_marginal_relevance_search.return_value = [doc1, doc2]
    
    retriever = DocuMindRetriever(chroma_manager=mock_chroma, api_key="mock_key", use_rewriter=False, use_compressor=False)
    
    # Test Similarity Search
    res_sim = retriever.retrieve("query text", strategy="Similarity Search", k=2)
    assert res_sim["count"] == 2
    assert abs(res_sim["scores"][0] - 0.6667) < 0.001
    assert abs(res_sim["scores"][1] - 0.4000) < 0.001
    assert res_sim["chunks"][0].page_content == "Text chunk 1"
    
    # Test MMR Search
    res_mmr = retriever.retrieve("query text", strategy="MMR Search", k=2)
    assert res_mmr["count"] == 2
    assert len(res_mmr["scores"]) == 2
    
    # Test Score Threshold Search
    # With threshold = 0.5, doc2 (similarity 0.4) should be filtered out
    res_thresh = retriever.retrieve("query text", strategy="Score Threshold Search", k=2, score_threshold=0.5)
    assert res_thresh["count"] == 1
    assert res_thresh["chunks"][0].metadata["chunk_id"] == "id1"
