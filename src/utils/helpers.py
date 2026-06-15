import json
from typing import List, Dict, Any
from fpdf import FPDF
from src.utils.logger import logger

def calculate_llm_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """
    Calculates the API cost based on OpenAI's pricing for the specified model.
    
    GPT-4o-mini rates:
    - Input: $0.150 per 1M tokens ($0.00000015 per token)
    - Output: $0.600 per 1M tokens ($0.00000060 per token)
    
    text-embedding-3-small rates:
    - $0.02 per 1M tokens ($0.00000002 per token)
    """
    if model == "gpt-4o-mini":
        input_rate = 0.15 / 1_000_000
        output_rate = 0.60 / 1_000_000
        return (prompt_tokens * input_rate) + (completion_tokens * output_rate)
    elif model == "text-embedding-3-small":
        # Usually prompt_tokens matches the tokens embedded
        rate = 0.02 / 1_000_000
        return prompt_tokens * rate
    return 0.0

def format_size(bytes_size: int) -> str:
    """Formats bytes size to a readable string (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def export_chat_to_pdf(chat_history: List[Dict[str, Any]]) -> bytes:
    """
    Exports chat history into a formatted PDF byte stream.
    Replaces non-latin1 characters to avoid encoding issues with default FPDF fonts.
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
            pdf.set_font("helvetica", "B", 10)
            if role == "USER":
                pdf.set_text_color(79, 70, 229)  # indigo-600
                pdf.cell(0, 6, f"User:", ln=True)
            elif role == "ASSISTANT":
                pdf.set_text_color(13, 148, 136)  # teal-600
                pdf.cell(0, 6, f"DocuMind AI:", ln=True)
            else:
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 6, f"{role}:", ln=True)
            
            # Message Body
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(51, 65, 85)  # slate-700
            
            # Sanitize content for default Latin-1 encoding
            sanitized_content = content.encode("latin-1", "replace").decode("latin-1")
            
            pdf.multi_cell(0, 5, sanitized_content)
            pdf.ln(5)
            
        return pdf.output()
    except Exception as e:
        logger.error(f"Failed to generate conversation PDF: {e}")
        raise e

def export_chat_to_json(chat_history: List[Dict[str, Any]]) -> str:
    """Serializes chat history list to a formatted JSON string."""
    return json.dumps(chat_history, indent=2, ensure_ascii=False)
