import json
from typing import List, Dict, Any
from fpdf import FPDF
from src.utils.logger import logger

def sanitize_text(text: str) -> str:
    """
    Sanitizes text to prevent FPDF encoding crashes by encoding to Latin-1
    and replacing incompatible characters.
    """
    if not text:
        return ""
    return text.encode("latin-1", "replace").decode("latin-1")

def export_to_txt(chat_history: List[Dict[str, Any]]) -> str:
    """
    Converts conversation history into a formatted TXT transcript,
    including questions, answers, and detailed sources.
    """
    lines = []
    lines.append("==================================================")
    lines.append("              DocuMind AI Chat Transcript        ")
    lines.append("==================================================")
    lines.append("")
    
    for idx, msg in enumerate(chat_history):
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        
        if role == "USER":
            lines.append(f"USER:")
            lines.append(content)
            lines.append("-" * 50)
        elif role == "ASSISTANT":
            lines.append(f"DOCUMIND AI:")
            lines.append(content)
            
            # Print sources if present
            sources = msg.get("sources", [])
            if sources:
                lines.append("")
                lines.append("SOURCES RETRIEVED:")
                for src_idx, src in enumerate(sources):
                    lines.append(
                        f"  [{src_idx+1}] File: {src['filename']} | Chunk ID: {src['chunk_id']} | "
                        f"Similarity: {src['score']:.4f}"
                    )
                    # Show preview content of source chunk
                    lines.append(f"  Content: \"{src['content'].strip()}\"")
                    lines.append("")
            lines.append("=" * 50)
        else:
            lines.append(f"{role}:")
            lines.append(content)
            lines.append("-" * 50)
        lines.append("")
        
    return "\n".join(lines)

def export_to_markdown(chat_history: List[Dict[str, Any]]) -> str:
    """
    Converts conversation history into a beautifully formatted Markdown transcript,
    using blockquotes for source chunks.
    """
    lines = []
    lines.append("# DocuMind AI Chat Transcript")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    for msg in chat_history:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        
        if role == "USER":
            lines.append(f"### 👤 User")
            lines.append(content)
            lines.append("")
        elif role == "ASSISTANT":
            lines.append(f"### 🤖 DocuMind AI")
            lines.append(content)
            lines.append("")
            
            # Formats sources list in markdown blockquotes
            sources = msg.get("sources", [])
            if sources:
                lines.append("#### 📚 Sources & Citations")
                for src_idx, src in enumerate(sources):
                    lines.append(
                        f"- **[{src_idx+1}] {src['filename']}** — Chunk: `{src['chunk_id']}` "
                        f"(Similarity Score: `{src['score']:.4f}`)"
                    )
                    # Quote chunk text cleanly
                    quoted_text = "\n".join([f"  > {l}" for l in src['content'].splitlines()])
                    lines.append(quoted_text)
                    lines.append("")
            lines.append("---")
            lines.append("")
        else:
            lines.append(f"### ⚙️ {role.title()}")
            lines.append(content)
            lines.append("")
            
    return "\n".join(lines)

def export_to_pdf(chat_history: List[Dict[str, Any]]) -> bytes:
    """
    Exports conversation history into a formatted PDF byte stream
    using FPDF, including colored roles and source blocks.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Header block
        pdf.set_font("helvetica", "B", 18)
        pdf.set_text_color(15, 23, 42)  # slate-900
        pdf.cell(0, 10, "DocuMind AI", ln=True, align="C")
        
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(100, 116, 139)  # slate-500
        pdf.cell(0, 6, "Intelligent Knowledge Assistant - Chat Transcript", ln=True, align="C")
        pdf.line(10, 30, 200, 30)
        pdf.ln(10)
        
        for idx, message in enumerate(chat_history):
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            
            # Message Metadata / Role
            pdf.set_font("helvetica", "B", 11)
            if role == "USER":
                pdf.set_text_color(79, 70, 229)  # Indigo-600
                pdf.cell(0, 7, f"User:", ln=True)
            elif role == "ASSISTANT":
                pdf.set_text_color(13, 148, 136)  # Teal-600
                pdf.cell(0, 7, f"DocuMind AI:", ln=True)
            else:
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 7, f"{role.capitalize()}:", ln=True)
            
            # Message Body
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(51, 65, 85)  # Slate-700
            
            sanitized_content = sanitize_text(content)
            pdf.multi_cell(0, 5, sanitized_content)
            pdf.ln(3)
            
            # Render sources if present for assistant responses
            sources = message.get("sources", [])
            if role == "ASSISTANT" and sources:
                pdf.set_font("helvetica", "B", 9)
                pdf.set_text_color(71, 85, 105)  # slate-600
                pdf.cell(0, 5, "Sources & Citations:", ln=True)
                pdf.ln(1)
                
                for src_idx, src in enumerate(sources):
                    pdf.set_font("helvetica", "B", 8)
                    pdf.set_text_color(100, 116, 139)
                    src_title = f"  [{src_idx+1}] File: {src['filename']} — Chunk: {src['chunk_id']} (Similarity: {src['score']:.4f})"
                    pdf.cell(0, 4, sanitize_text(src_title), ln=True)
                    
                    pdf.set_font("helvetica", "I", 8)
                    pdf.set_text_color(120, 130, 140)
                    # Render content slightly indented with left margin padding
                    pdf.set_x(15)
                    pdf.multi_cell(0, 4, sanitize_text(src['content'].strip()))
                    pdf.set_x(10)
                    pdf.ln(2)
                pdf.ln(2)
                
        return pdf.output()
    except Exception as e:
        logger.error(f"Failed to generate conversation PDF: {e}")
        raise e
