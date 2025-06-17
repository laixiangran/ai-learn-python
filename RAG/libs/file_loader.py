import os
from typing import Literal
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


class CustomMarkdownLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> str:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件 {self.file_path} 不存在。")
        with open(self.file_path, "r", encoding="utf-8") as file:
            content = file.read()
        document = Document(page_content=content, metadata={"source": self.file_path})
        return [document]


def FileLoader(
    file_path: str = None,
    provider: Literal[
        "customMarkdownLoader", "unstructuredMarkdownLoader"
    ] = "customMarkdownLoader",
    **kwargs,
):
    if provider == "customMarkdownLoader":
        return CustomMarkdownLoader(file_path=file_path)
    elif provider == "unstructuredMarkdownLoader":
        return UnstructuredMarkdownLoader(file_path=file_path, **kwargs)
    else:
        raise ValueError(f"Provider {provider} not supported.")
