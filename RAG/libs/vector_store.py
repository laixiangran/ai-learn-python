import os
from typing import Any, Literal
from langchain_chroma import Chroma


def VectorStore(
    collection_name="my_collection",
    embedding_function=None,
    persist_directory="./RAG/chroma_data",
    provider: Literal["chroma"] = "chroma",
    **kwargs: Any,
):
    if provider == "chroma":
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            **kwargs,
        )
    else:
        raise ValueError(f"Provider {provider} not supported.")
