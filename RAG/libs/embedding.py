import os
from typing import Literal
from langchain_ollama import OllamaEmbeddings


def Embedding(model: str = None, provider: Literal["ollama"] = "ollama", **kwargs):
    if provider == "ollama":
        return OllamaEmbeddings(model=model, **kwargs)
    else:
        raise ValueError(f"Provider {provider} not supported.")
