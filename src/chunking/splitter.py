from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.ingestion.parser import MarkdownParser
from src.utils.logger import logger

class MarkdownSplitter:
    """
    Cleans and splits Markdown files into chunks of configurable size.
    Annotates each chunk with precise metadata including source file,
    chunk index, and nearest preceding section header.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parser = MarkdownParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def split_document(self, doc: Dict[str, Any]) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Cleans, processes, and splits a single document dict into chunked Langchain Documents.
        
        Args:
            doc (Dict[str, Any]): Dictionary loaded via DocumentLoader containing:
              - 'content': Raw Markdown string
              - 'path': Absolute file path
              - 'filename': Base file name
              - 'file_hash': SHA-256 content hash
              - 'timestamp': Date string of indexing
              
        Returns:
            Tuple[List[Document], Dict[str, Any]]:
              - List of langchain Documents with populated metadata.
              - Statistics dict with total, average, and maximum chunk sizes.
        """
        import re
        raw_content = doc.get("content", "")
        cleaned_content = self.parser.clean_text(raw_content)
        headers = self.parser.extract_headers(cleaned_content)

        # Split into sections by headers
        sections = []
        if not headers:
            sections.append({
                "title": "Document Header",
                "content": cleaned_content
            })
        else:
            first_header_start = headers[0]["start_index"]
            if first_header_start > 0:
                pre_text = cleaned_content[0:first_header_start].strip()
                if pre_text:
                    sections.append({
                        "title": "Document Header",
                        "content": pre_text
                    })
            
            for idx, header in enumerate(headers):
                start = header["start_index"]
                end = headers[idx+1]["start_index"] if idx + 1 < len(headers) else len(cleaned_content)
                section_text = cleaned_content[start:end]
                
                lines = section_text.splitlines()
                if lines and re.match(r'^#{1,6}\s+', lines[0]):
                    section_content = "\n".join(lines[1:])
                else:
                    section_content = section_text
                
                section_content = section_content.strip()
                if section_content:
                    sections.append({
                        "title": header["title"],
                        "content": section_content
                    })

        documents: List[Document] = []
        chunk_sizes: List[int] = []
        chunk_idx = 0

        for section in sections:
            section_title = section["title"]
            section_content = section["content"]
            
            # Split section content using RecursiveCharacterTextSplitter
            section_chunks = self.splitter.split_text(section_content)
            
            for chunk_text in section_chunks:
                chunk_id = f"{doc['file_hash']}_c{chunk_idx}"
                metadata = {
                    "chunk_id": chunk_id,
                    "chunk_position": chunk_idx,
                    "section_title": section_title,
                    "source": doc["path"],
                    "filename": doc["filename"],
                    "file_hash": doc["file_hash"],
                    "timestamp": doc["timestamp"]
                }
                documents.append(Document(page_content=chunk_text, metadata=metadata))
                chunk_sizes.append(len(chunk_text))
                chunk_idx += 1

        # Generate chunk statistics
        stats = {
            "total_chunks": len(documents),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0
        }

        logger.info(
            f"Chunked '{doc['filename']}' into {stats['total_chunks']} pieces. "
            f"Avg size: {stats['avg_chunk_size']:.1f} characters, Max size: {stats['max_chunk_size']}."
        )

        return documents, stats
