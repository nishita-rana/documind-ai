import os
import json
import time
from typing import List, Dict, Any
from src.utils.logger import logger

class MetricsTracker:
    """
    Evaluates and records performance metrics for RAG queries. Saves data to
    a local JSON history file to support long-term visualization and diagnostics.
    """
    def __init__(self, history_file: str = "logs/metrics_history.json"):
        """
        Initializes the MetricsTracker.
        
        Args:
            history_file (str): Location of metrics history JSON file.
        """
        self.history_file = os.path.abspath(history_file)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Creates parent directory and empty history list if they don't exist."""
        log_dir = os.path.dirname(self.history_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        if not os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            except Exception as e:
                logger.error(f"Error creating metrics history file '{self.history_file}': {e}")

    def load_history(self) -> List[Dict[str, Any]]:
        """Loads and returns all recorded query run metrics from history."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metrics history: {e}")
            return []

    def log_run(
        self,
        query: str,
        retrieved_count: int,
        compressed_count: int,
        retrieval_latency: float,
        generation_latency: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost: float,
        avg_similarity_score: float = 0.0,
        vector_store_size: int = 0,
        total_documents_indexed: int = 0,
        scores: List[float] = None,
        original_query: str = "",
        rewritten_query: str = "",
        similarity_improvement: float = 0.0,
        strategy: str = "Similarity Search"
    ) -> Dict[str, Any]:
        """
        Records search and inference metrics, appends them to disk history.
        
        Args:
            query (str): The search query.
            retrieved_count (int): Count of source chunks retrieved.
            compressed_count (int): Count of chunks after compression.
            retrieval_latency (float): Time spent in rewriter + search + compressor.
            generation_latency (float): Time spent in streaming generation.
            prompt_tokens (int): Count of input tokens.
            completion_tokens (int): Count of output tokens.
            total_tokens (int): Total token usage.
            cost (float): API dollar cost for the operation.
            avg_similarity_score (float): Average similarity score of retrieved chunks.
            vector_store_size (int): Size of the vector database in chunks.
            total_documents_indexed (int): Number of unique documents in the database.
            scores (List[float]): List of similarity scores.
            original_query (str): Original search query.
            rewritten_query (str): Standalone query used for retrieval.
            similarity_improvement (float): Improvement in similarity score compared to original query.
            strategy (str): Retrieval strategy used.
            
        Returns:
            Dict[str, Any]: Logged metrics details.
        """
        run_data = {
            "timestamp": time.time(),
            "query": query,
            "retrieved_chunks": retrieved_count,
            "compressed_chunks": compressed_count,
            "retrieval_latency": round(retrieval_latency, 4),
            "generation_latency": round(generation_latency, 4),
            "total_latency": round(retrieval_latency + generation_latency, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(cost, 6),
            "avg_similarity_score": round(avg_similarity_score, 4),
            "vector_store_size": vector_store_size,
            "total_documents_indexed": total_documents_indexed,
            "scores": scores or [],
            "original_query": original_query or query,
            "rewritten_query": rewritten_query or query,
            "similarity_improvement": round(similarity_improvement, 4),
            "strategy": strategy
        }

        history = self.load_history()
        history.append(run_data)

        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            logger.info(
                f"Logged RAG run metrics. Latency: {run_data['total_latency']:.4f}s. "
                f"Cost: ${run_data['cost']:.6f}"
            )
        except Exception as e:
            logger.error(f"Failed to save run metrics: {e}")

        return run_data

    def clear_history(self) -> None:
        """Clears all historical metrics logs."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info("Metrics history cleared.")
        except Exception as e:
            logger.error(f"Error clearing metrics history: {e}")
