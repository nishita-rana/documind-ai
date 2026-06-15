import os
import shutil
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.embeddings.embedding_service import OpenAIEmbeddingService
from src.utils.logger import logger

class ChromaManager:
    """
    Manages SQLite-backed Chroma vector store. Supports document persistence,
    document insertion, file-based deletion (to support incremental builds),
    collection reset, and metadata indexing statistics.
    """
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "documind_collection", embedding_service: OpenAIEmbeddingService = None):
        """
        Initializes the ChromaManager.
        
        Args:
            persist_dir (str): Relative or absolute path to database persistence.
            collection_name (str): Collection identifier.
            embedding_service (OpenAIEmbeddingService): Service for converting chunks to vectors.
        """
        self.persist_dir = os.path.abspath(persist_dir)
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.embeddings = embedding_service.get_embeddings_instance() if embedding_service else None
        
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> Chroma:
        """Creates or loads the LangChain Chroma wrapper."""
        logger.info(f"Initializing ChromaDB connection at '{self.persist_dir}', collection: '{self.collection_name}'")
        os.makedirs(self.persist_dir, exist_ok=True)
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def update_api_key(self, api_key: str) -> None:
        """
        Dynamically updates the embedding function API key.
        Allows the user to adjust credentials via the Streamlit UI.
        
        Args:
            api_key (str): The new OpenAI API key.
        """
        logger.info("Updating ChromaManager embedding service credentials.")
        self.embedding_service = OpenAIEmbeddingService(api_key=api_key)
        self.embeddings = self.embedding_service.get_embeddings_instance()
        self.vector_store = self._init_vector_store()

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds a list of LangChain Documents to ChromaDB.
        
        Args:
            documents (List[Document]): Document chunks to store.
        """
        if not documents:
            logger.info("No documents provided to add to vector store.")
            return

        logger.info(f"Inserting {len(documents)} document chunks into ChromaDB...")
        try:
            # Langchain handles batch ids internally or we can supply them.
            # Supplying clean IDs makes lookup and retrieval cleaner.
            ids = [doc.metadata.get("chunk_id", f"chk_{i}") for i, doc in enumerate(documents)]
            self.vector_store.add_documents(documents, ids=ids)
            logger.info(f"Successfully inserted {len(documents)} chunks.")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise e

    def delete_documents_by_file(self, file_path: str) -> None:
        """
        Queries all chunk IDs matching the specified source file path and deletes them.
        This enables clean, non-destructive file updates during incremental scanning.
        
        Args:
            file_path (str): File path of the documents to remove.
        """
        abs_path = os.path.abspath(file_path)
        logger.info(f"Searching and deleting existing vectors for file: '{abs_path}'")
        try:
            # Filter matches by exact absolute file path metadata
            results = self.vector_store.get(where={"source": abs_path})
            ids = results.get("ids", [])
            
            if ids:
                self.vector_store.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunks associated with file: '{abs_path}'")
            else:
                logger.info(f"No existing chunks found in database for file: '{abs_path}'")
        except Exception as e:
            logger.error(f"Failed to delete document vectors for file '{abs_path}': {e}")
            raise e

    def clear_database(self) -> None:
        """
        Wipes the current collection completely from vector storage.
        Falls back to directory removal if database process locking occurs.
        """
        logger.info("Clearing Chroma database collection.")
        try:
            # Wipe using LangChain
            self.vector_store.delete_collection()
            logger.info("ChromaDB collection deleted successfully.")
        except Exception as e:
            logger.warning(f"Failed to clear collection via API: {e}. Attempting file-level wipe.")
            self.vector_store = None
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                logger.info(f"Deleted persistent directory '{self.persist_dir}'")
        
        # Re-initialize vector store
        self.vector_store = self._init_vector_store()

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Gathers database health, chunk tallies, and catalog metrics.
        
        Returns:
            Dict[str, Any]: Database status metrics dictionary.
        """
        stats = {
            "total_chunks": 0,
            "unique_files_count": 0,
            "indexed_files": {},
            "status": "Uninitialized"
        }
        
        try:
            count = self.vector_store._collection.count()
            stats["total_chunks"] = count
            stats["status"] = "Healthy" if count > 0 else "Empty"

            if count > 0:
                results = self.vector_store.get(include=["metadatas"])
                metadatas = results.get("metadatas", [])
                
                indexed_files = {}
                for meta in metadatas:
                    if not meta:
                        continue
                    source = meta.get("source")
                    file_hash = meta.get("file_hash")
                    timestamp = meta.get("timestamp")
                    filename = meta.get("filename")

                    if source:
                        if source not in indexed_files:
                            indexed_files[source] = {
                                "filename": filename,
                                "file_hash": file_hash,
                                "timestamp": timestamp,
                                "chunk_count": 0
                            }
                        indexed_files[source]["chunk_count"] += 1
                
                stats["unique_files_count"] = len(indexed_files)
                stats["indexed_files"] = indexed_files
                
        except Exception as e:
            logger.error(f"Error querying collection stats: {e}")
            stats["status"] = f"Error: {str(e)}"
            
        return stats
