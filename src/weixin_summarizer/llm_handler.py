import os
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
# todo: what are differences between this two? Should I use ChatOllama? Which one is better?
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from tenacity import retry, stop_after_attempt, wait_exponential

# todo: is there any way to unify this? instead of having it everywhere?
from dotenv import load_dotenv
load_dotenv()

# todo: but really, there is no RAG implemented in this app yet. What is RAG? Together with Embedding? Anything else?


class LLMHandler:
    """Manages LLM interactions with support for different providers."""

    # todo: those environment variables can re-considered.
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o")

        self.provider = os.getenv("LLM_PROVIDER", "openai")
        # self.provider = os.getenv("LLM_PROVIDER", "ollama")

        self.llm = self._initialize_llm()
        
    def _initialize_llm(self) -> BaseLanguageModel:
        """Initializes the LLM based on configuration."""
        # todo: callbacks. consider using a callbacks.
        callbacks = []

        if os.getenv("LANGCHAIN_API_KEY"):
            print("LANGCHAIN_API_KEY found. Configure custom callbacks as needed.")

        # todo: in the current prompt setup, `System`, `Human` message are not specified.
        if self.provider == "openai":
            # todo: double check those parameters
            return ChatOpenAI(
                model=self.model_name,
                temperature=0.7,
                callbacks=callbacks,
                request_timeout=60,
            )
        elif self.provider == "ollama":
            return ChatOllama(
                model="qwen2:7b",
                temperature=0.8,
                num_predict=256,
            )
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