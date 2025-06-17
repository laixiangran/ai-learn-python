from typing import Literal
from langchain_ollama import ChatOllama


def Chat(model: str, provider: Literal["ollama"] = "ollama", **kwargs):
    if provider == "ollama":
        return ChatOllama(model=model, **kwargs)
    else:
        raise ValueError(f"Provider {provider} not supported.")
