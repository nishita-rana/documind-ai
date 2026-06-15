import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
from src.utils.logger import logger

class DocumentLoader:
    """
    Handles scanning directories recursively, checking file hashes for incremental indexing,
    and identifying added, modified, or deleted files relative to a manifest database.
    """
    def __init__(self, docs_dir: str = "docs", manifest_path: str = "chroma_db/index_manifest.json"):
        self.docs_dir = os.path.abspath(docs_dir)
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads indexing manifest from file or returns an empty dictionary."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading manifest file {self.manifest_path}: {e}")
        return {}

    def save_manifest(self) -> None:
        """Saves current manifest state to a JSON file."""
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        try:
            with open(self.manifest_path, 'w', encoding='utf-8') as f:
                json.dump(self.manifest, f, indent=2)
            logger.info(f"Manifest updated and saved. Total indexed files: {len(self.manifest)}")
        except Exception as e:
            logger.error(f"Error saving manifest file {self.manifest_path}: {e}")

    def compute_hash(self, content: str) -> str:
        """Computes SHA-256 hash of text contents."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def extract_pdf_text(self, file_path: str) -> str:
        """Extracts text content page-by-page from a PDF file."""
        from pypdf import PdfReader
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"# Page {i+1}\n{page_text}")
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from PDF '{file_path}': {e}")
            return ""

    def scan_directory(self) -> Tuple[List[Dict[str, Any]], Set[str], Set[str]]:
        """
        Scans docs_dir recursively for Markdown and PDF files.
        Detects duplicates by hashing contents.
        Compares with manifest to identify:
          - files that need indexing (new or modified)
          - files that were deleted from disk since last indexing
          - files that are unmodified
        
        Returns:
            Tuple[List[Dict[str, Any]], Set[str], Set[str]]: 
            - documents to index: list of dicts with keys: ['content', 'path', 'filename', 'file_hash', 'timestamp']
            - deleted files: set of absolute file paths
            - unmodified files: set of absolute file paths
        """
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            logger.info(f"Created docs directory: {self.docs_dir}")

        current_files: Dict[str, Tuple[str, str]] = {}
        
        # Walk directory recursively
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith('.md'):
                    full_path = os.path.abspath(os.path.join(root, file))
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        file_hash = self.compute_hash(content)
                        current_files[full_path] = (content, file_hash)
                    except Exception as e:
                        logger.error(f"Error reading file {full_path}: {e}")
                elif file.endswith('.pdf'):
                    full_path = os.path.abspath(os.path.join(root, file))
                    try:
                        content = self.extract_pdf_text(full_path)
                        if content:
                            file_hash = self.compute_hash(content)
                            current_files[full_path] = (content, file_hash)
                    except Exception as e:
                        logger.error(f"Error reading PDF file {full_path}: {e}")

        # Check for duplicates on disk
        seen_hashes: Dict[str, str] = {}
        unique_files: Dict[str, Tuple[str, str]] = {}
        for path, (content, file_hash) in current_files.items():
            if file_hash in seen_hashes:
                logger.warning(
                    f"Duplicate document detected! '{path}' shares identical content with "
                    f"'{seen_hashes[file_hash]}'. Skipping to prevent indexing duplicate vectors."
                )
            else:
                seen_hashes[file_hash] = path
                unique_files[path] = (content, file_hash)

        manifest_paths = set(self.manifest.keys())
        current_paths = set(unique_files.keys())

        # Determine states
        deleted_files = manifest_paths - current_paths
        unmodified_files = set()
        to_index_files = []

        for path in current_paths:
            content, file_hash = unique_files[path]
            # If path exists in manifest and the file hash matches, it's unmodified
            if path in self.manifest and self.manifest[path].get("file_hash") == file_hash:
                unmodified_files.add(path)
            else:
                filename = os.path.basename(path)
                timestamp = datetime.now().isoformat()
                to_index_files.append({
                    "content": content,
                    "path": path,
                    "filename": filename,
                    "file_hash": file_hash,
                    "timestamp": timestamp
                })

        logger.info(
            f"Directory scan completed. Found {len(to_index_files)} new/modified files, "
            f"{len(deleted_files)} deleted files, and {len(unmodified_files)} unmodified files."
        )

        return to_index_files, deleted_files, unmodified_files

    def update_manifest(self, file_path: str, file_hash: str) -> None:
        """Updates the local memory manifest dict for a specific file."""
        self.manifest[os.path.abspath(file_path)] = {
            "file_hash": file_hash,
            "last_indexed": datetime.now().isoformat()
        }

    def remove_from_manifest(self, file_path: str) -> None:
        """Removes a file from the local memory manifest dict."""
        abs_path = os.path.abspath(file_path)
        if abs_path in self.manifest:
            del self.manifest[abs_path]
            logger.info(f"Removed '{abs_path}' from index manifest.")

    def clear_manifest(self) -> None:
        """Clears all records in the manifest."""
        self.manifest.clear()
        self.save_manifest()
        logger.info("Manifest cleared and saved.")
