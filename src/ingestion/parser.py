import re
from typing import List, Dict, Any
from src.utils.logger import logger

class MarkdownParser:
    """
    Cleans markdown documents, normalizes whitespaces, and extracts headers
    to determine the section title context for each text chunk.
    """
    def __init__(self):
        # Matches markdown headings starting with #, ##, etc.
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def clean_text(self, text: str) -> str:
        """
        Cleans markdown text by removing trailing whitespaces and collapsing
        excessive newlines, while preserving code blocks and table layouts.
        
        Args:
            text (str): Raw Markdown text content.
            
        Returns:
            str: Cleaned Markdown text.
        """
        if not text:
            return ""

        # Normalize line endings and strip trailing spaces from lines
        lines = [line.rstrip() for line in text.splitlines()]
        trimmed_text = "\n".join(lines)

        # Normalize multiple newlines to at most two consecutive newlines
        cleaned_text = re.sub(r'\n{3,}', '\n\n', trimmed_text)
        return cleaned_text

    def extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts headers and their absolute character offset position in the text.
        
        Args:
            text (str): Cleaned markdown text.
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing level, title, and start_index.
        """
        headers = []
        for match in self.header_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_index = match.start()
            headers.append({
                "level": level,
                "title": title,
                "start_index": start_index
            })
        return headers

    @staticmethod
    def get_section_for_offset(headers: List[Dict[str, Any]], offset: int) -> str:
        """
        Finds the closest preceding section header for a given text character offset.
        
        Args:
            headers (List[Dict[str, Any]]): List of extracted headers with start indices.
            offset (int): Offset position of a chunk.
            
        Returns:
            str: Preceding header title or 'Document Header' if none exists.
        """
        current_section = "Document Header"
        for header in headers:
            if header["start_index"] <= offset:
                current_section = header["title"]
            else:
                break
        return current_section
