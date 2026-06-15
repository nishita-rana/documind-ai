import os
import tiktoken
from typing import List, Dict, Any, Generator
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from src.utils.logger import logger
from src.utils.helpers import calculate_llm_cost

class LLMGenerator:
    """
    Manages the response generation layer. Constructs prompts using system rules
    and context documents, feeds conversational state, streams response tokens,
    and computes API cost statistics.
    """
    def __init__(self, api_key: str = None, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Initializes the LLMGenerator.
        
        Args:
            api_key (str): Optional OpenAI API key.
            provider (str): The model provider ('openai' or 'ollama').
            model (str): The model name.
        """
        self.api_key = api_key
        self.provider = provider
        self.model_name = model
        self.llm = None
        
        self._init_llm(api_key)
        
        # Load tiktoken encoder for cl100k_base
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load cl100k_base encoder: {e}")
            self._encoder = tiktoken.get_encoding("gpt2")

    def _init_llm(self, api_key: str = None) -> None:
        """Helper to create LLM instance based on provider."""
        if self.provider == "openai":
            if api_key or "OPENAI_API_KEY" in os.environ:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=0.0,
                    openai_api_key=api_key,
                    streaming=True
                )
        elif self.provider == "ollama":
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0.0
            )

    def update_api_key(self, api_key: str) -> None:
        """Updates LLM credentials dynamically."""
        logger.info("Updating LLMGenerator credentials.")
        self.api_key = api_key
        self._init_llm(api_key)

    def count_tokens(self, text: str) -> int:
        """Returns the token length of a given text."""
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def generate_response(
        self,
        query: str,
        context_documents: List[Document],
        chat_history: List[Dict[str, str]]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generates a streaming response for the user's query utilizing context documents.
        Yields intermediate text parts, and finally returns a dictionary containing metrics.
        
        Args:
            query (str): Standalone retrieval query.
            context_documents (List[Document]): Context documents matching the search terms.
            chat_history (List[Dict[str, str]]): List of previous messages in the session.
            
        Yields:
            Dict[str, Any]: Either type 'text' with message contents or 'metrics' with statistics.
        """
        if not self.llm:
            yield {
                "type": "text",
                "content": "Error: OpenAI API key is missing. Please configure it in the sidebar or environment settings."
            }
            return

        # Format context chunks nicely
        context_sections = []
        for doc in context_documents:
            filename = doc.metadata.get("filename", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "Unknown")
            section = doc.metadata.get("section_title", "Unknown")
            
            context_sections.append(
                f"--- SOURCE: {filename} | CHUNK_ID: {chunk_id} | SECTION: {section} ---\n"
                f"{doc.page_content}\n"
                f"--------------------------------------------------"
            )
        
        context_str = "\n\n".join(context_sections)

        # Build prompt template
        system_content = (
            "You are a highly helpful and precise document assistant named DocuMind AI.\n\n"
            "Rules:\n"
            "1. Answer the question ONLY using the retrieved context provided below.\n"
            "2. Do NOT make up facts or extrapolate beyond the context. Never hallucinate.\n"
            "3. If the retrieved context does not contain enough information to answer the question, say exactly:\n"
            "   \"The provided documents do not contain enough information.\"\n"
            "4. Always cite your sources inside the text (e.g. [source_file.md, Chunk ID: xyz]) when referring to facts from a specific chunk.\n"
            "5. Be concise, professional, and accurate.\n\n"
            f"Retrieved Context:\n{context_str}"
        )

        messages = [SystemMessage(content=system_content)]

        # Bind previous chat history messages (limit context window size)
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Add the current user query
        messages.append(HumanMessage(content=query))

        # Estimate Prompt Tokens
        raw_prompt_text = system_content + "\n".join([m.content for m in messages[1:]])
        prompt_tokens = self.count_tokens(raw_prompt_text) + (len(messages) * 4)  # Small overhead padding
        
        logger.info(f"Invoking LLM streaming. Estimated prompt tokens: {prompt_tokens}")

        full_response_text = ""
        try:
            for chunk in self.llm.stream(messages):
                text_part = chunk.content
                full_response_text += text_part
                yield {"type": "text", "content": text_part}

            # Generate final generation cost and token usage statistics
            completion_tokens = self.count_tokens(full_response_text)
            total_tokens = prompt_tokens + completion_tokens
            cost = calculate_llm_cost(prompt_tokens, completion_tokens, model="gpt-4o-mini") if self.provider == "openai" else 0.0

            logger.info(
                f"LLM Stream completed. Completion tokens: {completion_tokens}. "
                f"Estimated response cost: ${cost:.6f}"
            )

            yield {
                "type": "metrics",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost
            }

        except Exception as e:
            logger.error(f"Error during streaming execution: {e}")
            yield {
                "type": "text",
                "content": f"\n\n[Error occurred during LLM stream: {str(e)}]"
            }
