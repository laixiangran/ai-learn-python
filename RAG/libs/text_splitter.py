from typing import Literal
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)


def TextSplitter(
    provider: Literal[
        "recursiveCharacterTextSplitter", "markdownTextSplitter"
    ] = "recursiveCharacterTextSplitter",
    **kwargs,
):
    if provider == "recursiveCharacterTextSplitter":
        return RecursiveCharacterTextSplitter(**kwargs)
    elif provider == "markdownTextSplitter":
        return MarkdownTextSplitter(**kwargs)
    else:
        raise ValueError(f"Provider {provider} not supported.")
