import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore

class RetrievalStrategy(ABC):
    """
    Abstract Base Class defining the interface for document retrieval strategies
    from the vector database.
    """
    @abstractmethod
    def retrieve(
        self,
        vector_store: VectorStore,
        query: str,
        k: int,
        score_threshold: float,
        convert_score_fn: Any
    ) -> Tuple[List[Document], List[float]]:
        """
        Executes document retrieval using a specific search strategy.

        Args:
            vector_store (VectorStore): The active vector database store instance.
            query (str): The search-optimized query query.
            k (int): Number of top documents to retrieve.
            score_threshold (float): Similarity threshold for filtering.
            convert_score_fn (callable): Mapping function from raw distance to similarity score.

        Returns:
            Tuple[List[Document], List[float]]: A tuple containing the list of retrieved
            documents and their corresponding similarity scores.
        """
        pass

class SimilaritySearchStrategy(RetrievalStrategy):
    """
    Retrieves documents using standard semantic similarity search based on L2 distance.
    """
    def retrieve(
        self,
        vector_store: VectorStore,
        query: str,
        k: int,
        score_threshold: float,
        convert_score_fn: Any
    ) -> Tuple[List[Document], List[float]]:
        original_chunks = []
        scores = []
        
        sim_results = vector_store.similarity_search_with_score(query, k=k)
        for doc, dist in sim_results:
            original_chunks.append(doc)
            scores.append(convert_score_fn(dist))
            
        return original_chunks, scores

class MMRSearchStrategy(RetrievalStrategy):
    """
    Retrieves diverse documents using Max Marginal Relevance (MMR) search.
    Maximizes relevance to the query while minimizing redundancy between chunks.
    """
    def retrieve(
        self,
        vector_store: VectorStore,
        query: str,
        k: int,
        score_threshold: float,
        convert_score_fn: Any
    ) -> Tuple[List[Document], List[float]]:
        fetch_k = max(20, k * 2)
        raw_docs = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        original_chunks = raw_docs
        
        # Retrieve similarity scores for these documents by running a similarity search
        sim_results = vector_store.similarity_search_with_score(query, k=fetch_k)
        doc_score_map = {}
        for doc, dist in sim_results:
            doc_id = doc.metadata.get("chunk_id", doc.page_content)
            doc_score_map[doc_id] = convert_score_fn(dist)
            
        scores = []
        for doc in original_chunks:
            doc_id = doc.metadata.get("chunk_id", doc.page_content)
            scores.append(doc_score_map.get(doc_id, 0.5))
            
        return original_chunks, scores

class ScoreThresholdSearchStrategy(RetrievalStrategy):
    """
    Retrieves documents based on semantic similarity, filtering out any chunks
    that score below a specific minimum relevance threshold.
    """
    def retrieve(
        self,
        vector_store: VectorStore,
        query: str,
        k: int,
        score_threshold: float,
        convert_score_fn: Any
    ) -> Tuple[List[Document], List[float]]:
        original_chunks = []
        scores = []
        
        sim_results = vector_store.similarity_search_with_score(query, k=k * 2)
        for doc, dist in sim_results:
            sim_score = convert_score_fn(dist)
            if sim_score >= score_threshold:
                original_chunks.append(doc)
                scores.append(sim_score)
            if len(original_chunks) >= k:
                break
                
        return original_chunks, scores
