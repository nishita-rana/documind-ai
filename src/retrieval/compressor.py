import os
from typing import List
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from src.utils.logger import logger

class ContextCompressor:
    """
    Compresses retrieved text chunks using an LLM to extract only the sentences
    relevant to the user query. This reduces token consumption in LLM generation.
    """
    def __init__(self, api_key: str = None, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Initializes the ContextCompressor.
        
        Args:
            api_key (str): Optional OpenAI API key.
            provider (str): The model provider ('openai' or 'ollama').
            model (str): The model name.
        """
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.llm = None
        self.compressor = None
        
        self._init_compressor(api_key)

    def _init_compressor(self, api_key: str = None) -> None:
        """Helper to instantiate LLM and LangChain extractor."""
        if self.provider == "openai":
            if api_key or "OPENAI_API_KEY" in os.environ:
                self.llm = ChatOpenAI(
                    model=self.model,
                    temperature=0.0,
                    openai_api_key=api_key
                )
        elif self.provider == "ollama":
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=self.model,
                temperature=0.0
            )
        
        if self.llm:
            self.compressor = LLMChainExtractor.from_llm(self.llm)

    def update_api_key(self, api_key: str) -> None:
        """Updates LLM credentials dynamically."""
        logger.info("Updating ContextCompressor credentials.")
        self.api_key = api_key
        self._init_compressor(api_key)

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Compresses each document text chunk down to contextually relevant statements.
        
        Args:
            documents (List[Document]): List of retrieved chunks.
            query (str): The search query.
            
        Returns:
            List[Document]: List of compressed chunks, preserving original metadata.
        """
        if not self.compressor:
            logger.warning("ContextCompressor LLM or extractor not initialized. Returning raw documents.")
            return documents
            
        if not documents:
            return []

        logger.info(f"Compressing {len(documents)} source chunks against search query: '{query}'")
        try:
            compressed_docs = self.compressor.compress_documents(documents, query)
            logger.info(f"Successfully compressed context. Result count: {len(compressed_docs)} chunks.")
            
            # Re-verify and restore metadata links just in case LangChain strips any properties
            for orig, comp in zip(documents, compressed_docs):
                # Only overwrite missing metadata keys to retain chunk details
                for key, val in orig.metadata.items():
                    if key not in comp.metadata:
                        comp.metadata[key] = val
                        
            return compressed_docs
        except Exception as e:
            logger.error(f"Error during context compression: {e}. Falling back to uncompressed documents.")
            return documents
