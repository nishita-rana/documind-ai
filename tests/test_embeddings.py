import pytest
from unittest.mock import MagicMock, patch
from src.embeddings.embedding_service import OpenAIEmbeddingService

def test_token_counter():
    service = OpenAIEmbeddingService(api_key="mock_key")
    text = "Hello, world! This is a test."
    tokens = service.count_tokens(text)
    assert tokens > 0
    
    texts = ["Hello world", "This is another test sentence"]
    assert service.count_tokens_list(texts) == service.count_tokens(texts[0]) + service.count_tokens(texts[1])

@patch("src.embeddings.embedding_service.OpenAIEmbeddings")
def test_embed_documents_mocked(mock_embeddings_cls):
    # Setup mock instance
    mock_instance = MagicMock()
    mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_embeddings_cls.return_value = mock_instance
    
    # Initialize service
    service = OpenAIEmbeddingService(api_key="mock_key", batch_size=2)
    texts = ["Doc 1 content", "Doc 2 content"]
    
    vectors = service.embed_documents(texts)
    
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]
    
    # Verify that the underlying embed_documents was called
    mock_instance.embed_documents.assert_called_once_with(texts)
