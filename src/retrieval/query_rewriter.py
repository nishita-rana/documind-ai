from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.logger import logger

class QueryRewriter:
    """
    Analyzes user questions in the context of recent chat history and rewrites
    them into standalone search queries to optimize vector database retrieval.
    """
    def __init__(self, api_key: str = None, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Initializes the QueryRewriter.
        
        Args:
            api_key (str): Optional OpenAI API Key.
            provider (str): The model provider ('openai' or 'ollama').
            model (str): The model name.
        """
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.llm = None
        self._init_llm(api_key)

    def _init_llm(self, api_key: str = None) -> None:
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

    def update_api_key(self, api_key: str) -> None:
        """Dynamically reconfigures the LLM instance with new credentials."""
        self.api_key = api_key
        self._init_llm(api_key)

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]], force_rewrite: bool = False) -> str:
        """
        Rewrites a conversational query based on chat history.
        Skips LLM call if chat history is empty and force_rewrite is False to minimize latency and token cost.
        
        Args:
            query (str): The raw follow-up question from the user.
            chat_history (List[Dict[str, str]]): List of previous messages in the session.
            force_rewrite (bool): If True, forces the LLM to rewrite the query (e.g., expanding vague queries) even if history is empty.
            
        Returns:
            str: Standalone query optimized for document retrieval.
        """
        if not chat_history and not force_rewrite:
            logger.debug("Chat history is empty and force_rewrite is False. Skipping query rewriting.")
            return query

        if not self.llm:
            logger.warning("QueryRewriter LLM not initialized (missing API key). Returning raw query.")
            return query

        if not chat_history:
            logger.info(f"Rewriting user query without history: '{query}' (force_rewrite=True).")
            prompt_template = ChatPromptTemplate.from_template(
                "You are an expert search query optimizer for a document retrieval system (RAG) containing resumes, CVs, and professional documents.\n"
                "Your task is to rewrite the user's input query to be a standalone, search-optimized query targeting professional documents.\n"
                "If the user query is vague or very brief (e.g., 'What about education?', 'experience', 'what is experience'), expand it to ask a specific, professional, and clear question targeting the document collection.\n\n"
                "Examples:\n"
                "Input: 'What about education?' -> Output: 'What educational qualifications are present in the uploaded documents?'\n"
                "Input: 'What is experience?' -> Output: 'What professional experience is listed in the uploaded documents?'\n\n"
                "Do NOT add any conversational introduction, quotes, explanations, or system replies. "
                "Output ONLY the final rewritten search query.\n\n"
                "Input Query: {query}\n"
                "Standalone Query:"
            )
            try:
                prompt = prompt_template.format_messages(query=query)
                response = self.llm.invoke(prompt)
                rewritten_query = response.content.strip()
                if rewritten_query.startswith('"') and rewritten_query.endswith('"'):
                    rewritten_query = rewritten_query[1:-1].strip()
                logger.info(f"Rewritten query output (no history): '{rewritten_query}'")
                return rewritten_query
            except Exception as e:
                logger.error(f"Failed to rewrite query without history: {e}. Falling back to raw query.")
                return query

        logger.info(f"Rewriting user query: '{query}' with history of {len(chat_history)} messages.")
        
        # Format chat history for prompt consumption
        formatted_history = []
        for msg in chat_history[-6:]:  # Limit history context to last 3 turns
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted_history.append(f"{role}: {content}")
        history_str = "\n".join(formatted_history)

        prompt_template = ChatPromptTemplate.from_template(
            "Given the following conversation history and a user follow-up question, "
            "rewrite the follow-up question to be a standalone, search-optimized query "
            "that contains all necessary context for document retrieval.\n"
            "If the follow-up question is vague or very brief, expand it to ask a concrete, search-optimized question targeting the document collection (resumes, CVs, professional documents).\n\n"
            "Examples:\n"
            "Input Query: 'What about education?' -> Output: 'What educational qualifications are present in the uploaded documents?'\n"
            "Input Query: 'What is experience?' -> Output: 'What professional experience is listed in the uploaded documents?'\n\n"
            "Do NOT add any conversational introduction, quotes, explanations, or system replies. "
            "Output ONLY the final rewritten search query.\n\n"
            "Conversation History:\n"
            "{history_str}\n\n"
            "Follow-up Question: {query}\n"
            "Standalone Query:"
        )

        try:
            prompt = prompt_template.format_messages(
                history_str=history_str,
                query=query
            )
            response = self.llm.invoke(prompt)
            rewritten_query = response.content.strip()
            
            # Remove any surrounding double quotes that LLMs sometimes add
            if rewritten_query.startswith('"') and rewritten_query.endswith('"'):
                rewritten_query = rewritten_query[1:-1].strip()
                
            logger.info(f"Rewritten query output: '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}. Falling back to raw query.")
            return query

import os
