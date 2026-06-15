import os
import pytest
from src.ingestion.loader import DocumentLoader
from src.ingestion.parser import MarkdownParser

def test_markdown_parser_cleanup():
    parser = MarkdownParser()
    
    # Test whitespace normalization
    raw_text = "Hello World   \n\n\n\nNew Line   \n"
    cleaned = parser.clean_text(raw_text)
    assert cleaned == "Hello World\n\nNew Line"

def test_markdown_parser_headers():
    parser = MarkdownParser()
    text = "# Title\n\nSome intro text.\n\n## Section A\nDetail here.\n\n### Sub B\nMore details."
    
    headers = parser.extract_headers(text)
    assert len(headers) == 3
    assert headers[0]["title"] == "Title"
    assert headers[0]["level"] == 1
    assert headers[1]["title"] == "Section A"
    assert headers[1]["level"] == 2
    assert headers[2]["title"] == "Sub B"
    assert headers[2]["level"] == 3

    # Test offset resolution
    # Title offset: 0, Section A offset: 26, Sub B offset: 48
    sec1 = parser.get_section_for_offset(headers, 10)
    assert sec1 == "Title"
    
    sec2 = parser.get_section_for_offset(headers, 35)
    assert sec2 == "Section A"
    
    sec3 = parser.get_section_for_offset(headers, 60)
    assert sec3 == "Sub B"
    
    sec_first = parser.get_section_for_offset(headers, 0)
    assert sec_first == "Title"

def test_document_loader(tmp_path):
    # Setup folders
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"

    # Create dummy docs
    doc1 = docs_dir / "doc1.md"
    doc1.write_text("# Doc 1\nThis is the first document.", encoding="utf-8")
    
    # Duplicate of doc 1
    doc2 = docs_dir / "doc2.md"
    doc2.write_text("# Doc 1\nThis is the first document.", encoding="utf-8")
    
    # Unique doc
    doc3 = docs_dir / "doc3.md"
    doc3.write_text("# Doc 3\nThis is another unique document.", encoding="utf-8")

    loader = DocumentLoader(docs_dir=str(docs_dir), manifest_path=str(manifest_path))
    to_index, deleted, unmodified = loader.scan_directory()

    # Out of doc1 and doc2, one should be skipped as duplicate
    # So to_index should have exactly 2 documents (doc1/doc2, and doc3)
    assert len(to_index) == 2
    assert len(deleted) == 0
    assert len(unmodified) == 0

    # Index doc1 and doc3 in manifest
    loader.update_manifest(to_index[0]["path"], to_index[0]["file_hash"])
    loader.update_manifest(to_index[1]["path"], to_index[1]["file_hash"])
    loader.save_manifest()

    # Re-run loader. The 2 documents should be unmodified now
    loader2 = DocumentLoader(docs_dir=str(docs_dir), manifest_path=str(manifest_path))
    to_index2, deleted2, unmodified2 = loader2.scan_directory()
    assert len(to_index2) == 0
    assert len(unmodified2) == 2

    # Modify doc3
    doc3.write_text("# Doc 3\nModified text now.", encoding="utf-8")
    loader3 = DocumentLoader(docs_dir=str(docs_dir), manifest_path=str(manifest_path))
    to_index3, deleted3, unmodified3 = loader3.scan_directory()
    assert len(to_index3) == 1
    assert to_index3[0]["path"] == os.path.abspath(str(doc3))
    assert len(unmodified3) == 1

    # Delete doc1
    os.remove(doc1)
    loader4 = DocumentLoader(docs_dir=str(docs_dir), manifest_path=str(manifest_path))
    to_index4, deleted4, unmodified4 = loader4.scan_directory()
    assert len(deleted4) == 1
    assert list(deleted4)[0] == os.path.abspath(str(doc1))

from unittest.mock import patch

@patch("src.ingestion.loader.DocumentLoader.extract_pdf_text")
def test_pdf_document_loader(mock_extract_pdf, tmp_path):
    # Setup folders
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"

    # Create dummy pdf file path
    pdf_path = docs_dir / "test_doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock content")

    mock_extract_pdf.return_value = "# Page 1\nThis is page 1 content.\n\n# Page 2\nThis is page 2 content."

    loader = DocumentLoader(docs_dir=str(docs_dir), manifest_path=str(manifest_path))
    to_index, deleted, unmodified = loader.scan_directory()

    assert len(to_index) == 1
    assert to_index[0]["filename"] == "test_doc.pdf"
    assert to_index[0]["content"] == "# Page 1\nThis is page 1 content.\n\n# Page 2\nThis is page 2 content."
    
    # Verify the splitter breaks it into 2 chunks by page sections
    from src.chunking.splitter import MarkdownSplitter
    splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=50)
    chunks, stats = splitter.split_document(to_index[0])
    
    assert len(chunks) == 2
    assert chunks[0].metadata["section_title"] == "Page 1"
    assert chunks[0].page_content == "This is page 1 content."
    assert chunks[1].metadata["section_title"] == "Page 2"
    assert chunks[1].page_content == "This is page 2 content."

