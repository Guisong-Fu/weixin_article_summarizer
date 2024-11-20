import os
from typing import Optional, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import LangSmithCallbackHandler
from langchain.schema import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMHandler:
    """Manages LLM interactions with support for different providers."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.provider = os.getenv("LLM_PROVIDER", "openai")
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self) -> BaseLanguageModel:
        """Initializes the LLM based on configuration."""
        callbacks = []
        if os.getenv("LANGCHAIN_API_KEY"):
            callbacks.append(LangSmithCallbackHandler())
            
        if self.provider == "openai":
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=0.7,
                callbacks=callbacks,
                request_timeout=60,
            )
        elif self.provider == "ollama":
            # Add Ollama support here if needed
            raise NotImplementedError("Ollama support not yet implemented")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_completion(
        self, 
        prompt_template: str, 
        variables: Dict[str, Any]
    ) -> str:
        """Generates completion using the configured LLM."""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = await chain.ainvoke(variables)
        return response.content 