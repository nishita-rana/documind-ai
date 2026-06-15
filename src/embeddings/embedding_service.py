from typing import List
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from src.utils.logger import logger
from src.utils.helpers import calculate_llm_cost

class OpenAIEmbeddingService:
    """
    Service wrapper around OpenAI embedding APIs. Uses LangChain's OpenAIEmbeddings
    under the hood, adding token usage calculations, rate limit retries, logging,
    and batching controls.
    """
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small", batch_size: int = 250, provider: str = "openai"):
        """
        Initializes OpenAIEmbeddingService.
        
        Args:
            api_key (str): Optional OpenAI API key.
            model (str): Underlying model, defaults to 'text-embedding-3-small'.
            batch_size (int): Size of document batches sent to OpenAI.
            provider (str): The embedding provider ('openai' or 'ollama').
        """
        self.model = model
        self.batch_size = batch_size
        self.api_key = api_key
        self.provider = provider

        # Initialize the underlying langchain client
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key,
                max_retries=5,
                request_timeout=30.0,
                chunk_size=batch_size
            )
        elif provider == "ollama":
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text"
            )
        
        # Load tiktoken encoder for cl100k_base (compatible with modern models)
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load cl100k_base. Defaulting to gpt2 encoder: {e}")
            self._encoder = tiktoken.get_encoding("gpt2")

    def count_tokens(self, text: str) -> int:
        """Counts the token usage of a single string."""
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_tokens_list(self, texts: List[str]) -> int:
        """Counts the cumulative token usage of a list of strings."""
        return sum(self.count_tokens(text) for text in texts)

    def get_embeddings_instance(self):
        """Returns the underlying Langchain embeddings object."""
        return self.embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts. Operates in batches.
        
        Args:
            texts (List[str]): Input strings to embed.
            
        Returns:
            List[List[float]]: Matrix of text embeddings.
        """
        if not texts:
            return []

        total_tokens = self.count_tokens_list(texts)
        estimated_cost = calculate_llm_cost(total_tokens, 0, model="text-embedding-3-small") if self.provider == "openai" else 0.0
        logger.info(
            f"Embedding {len(texts)} chunks using {self.provider}. Total estimated tokens: {total_tokens}. "
            f"Estimated cost: ${estimated_cost:.6f}"
        )

        all_embeddings: List[List[float]] = []
        try:
            # Batch embedding collection manually to report progress and control rates
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                logger.info(f"Embedding batch {i // self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1} ({len(batch)} chunks)...")
                batch_vectors = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_vectors)

            logger.info("Successfully generated all text embeddings.")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error occurred during {self.provider} embeddings request: {e}")
            raise e
