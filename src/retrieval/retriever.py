import time
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from src.vectordb.chroma_manager import ChromaManager
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.compressor import ContextCompressor
from src.utils.logger import logger
from src.retrieval.strategies import (
    SimilaritySearchStrategy,
    MMRSearchStrategy,
    ScoreThresholdSearchStrategy
)

class DocuMindRetriever:
    """
    Orchestrates the retrieval layer. Integrates QueryRewriter and ContextCompressor
    with ChromaDB, supporting Similarity, MMR, and Score Threshold search strategies.
    Computes latency and formats retrieval outputs.
    """
    def __init__(
        self,
        chroma_manager: ChromaManager,
        api_key: str = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        use_rewriter: bool = True,
        use_compressor: bool = False
    ):
        """
        Initializes the retriever.
        
        Args:
            chroma_manager (ChromaManager): Connection wrapper for the database.
            api_key (str): OpenAI API key.
            provider (str): The model provider ('openai' or 'ollama').
            model (str): The model name.
            use_rewriter (bool): Flag to enable/disable query rewriting.
            use_compressor (bool): Flag to enable/disable sentence compression.
        """
        self.chroma_manager = chroma_manager
        self.use_rewriter = use_rewriter
        self.use_compressor = use_compressor
        
        self.query_rewriter = QueryRewriter(api_key=api_key, provider=provider, model=model)
        self.compressor = ContextCompressor(api_key=api_key, provider=provider, model=model)
        
        # Strategy pattern registry
        self.strategies = {
            "Similarity Search": SimilaritySearchStrategy(),
            "MMR Search": MMRSearchStrategy(),
            "Score Threshold Search": ScoreThresholdSearchStrategy()
        }

    def update_api_key(self, api_key: str) -> None:
        """Propagates API key updates down to the rewriter and compressor sub-services."""
        self.query_rewriter.update_api_key(api_key)
        self.compressor.update_api_key(api_key)

    def _convert_distance_to_similarity(self, distance: float) -> float:
        """
        Converts Chroma's raw L2 distance metrics to a standard 0-1 similarity scale.
        For L2 distance, similarity = 1 / (1 + distance) is robust.
        """
        return round(1.0 / (1.0 + distance), 4)

    def retrieve(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        strategy: str = "Similarity Search",
        k: int = 4,
        score_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes query rewriting, vector database query, and contextual compression.
        Measures total latency and captures scores.
        
        Args:
            query (str): The raw query input.
            chat_history (List[Dict[str, str]]): Conversational history.
            strategy (str): Retrieval technique ("Similarity Search", "MMR Search", "Score Threshold Search").
            k (int): Number of top documents to fetch.
            score_threshold (float): Similarity threshold (for Score Threshold strategy).
            
        Returns:
            Dict[str, Any]: Retrieval result bundle containing:
              - 'chunks': List[Document] (compressed if enabled, else original)
              - 'original_chunks': List[Document]
              - 'scores': List[float] (corresponding to original_chunks)
              - 'latency': float (in seconds)
              - 'count': int (number of final chunks)
              - 'query': str (search query used after possible rewriting)
        """
        start_time = time.time()
        
        # 1. Query Rewriting
        search_query = query
        if self.use_rewriter:
            try:
                search_query = self.query_rewriter.rewrite_query(query, chat_history or [], force_rewrite=True)
            except Exception as e:
                logger.error(f"Error during query rewriting: {e}. Using original query.")

        vector_store = self.chroma_manager.vector_store
        original_chunks: List[Document] = []
        scores: List[float] = []

        logger.info(f"Retrieving with strategy: '{strategy}', k={k}, query: '{search_query}'")

        # 2. Retrieval Execution
        try:
            strategy_obj = self.strategies.get(strategy)
            if not strategy_obj:
                logger.warning(f"Unknown retrieval strategy '{strategy}'. Defaulting to Similarity Search.")
                strategy_obj = self.strategies["Similarity Search"]
                
            original_chunks, scores = strategy_obj.retrieve(
                vector_store=vector_store,
                query=search_query,
                k=k,
                score_threshold=score_threshold,
                convert_score_fn=self._convert_distance_to_similarity
            )
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            original_chunks = []
            scores = []

        # 3. Context Compression
        compressed_chunks = original_chunks
        if self.use_compressor and original_chunks:
            try:
                compressed_chunks = self.compressor.compress_documents(original_chunks, search_query)
            except Exception as e:
                logger.error(f"Error during retrieval compression: {e}. Reverting to original chunks.")
                compressed_chunks = original_chunks

        # 4. Measure improvement if rewritten
        similarity_improvement = 0.0
        if self.use_rewriter and search_query != query:
            try:
                # Do a quick similarity search with the original query to measure baseline
                orig_results = vector_store.similarity_search_with_score(query, k=k)
                orig_scores = [self._convert_distance_to_similarity(dist) for _, dist in orig_results]
                orig_avg = sum(orig_scores) / len(orig_scores) if orig_scores else 0.0
                
                # Compare with rewritten average score
                new_avg = sum(scores) / len(scores) if scores else 0.0
                similarity_improvement = new_avg - orig_avg
            except Exception as e:
                logger.error(f"Error measuring retrieval improvement: {e}")

        latency = time.time() - start_time
        logger.info(f"Retrieved {len(compressed_docs := compressed_chunks)} final chunks in {latency:.4f} seconds.")

        return {
            "chunks": compressed_chunks,
            "original_chunks": original_chunks,
            "scores": scores,
            "latency": latency,
            "count": len(compressed_chunks),
            "query": search_query,
            "original_query": query,
            "rewritten_query": search_query,
            "similarity_improvement": similarity_improvement,
            "strategy": strategy
        }
