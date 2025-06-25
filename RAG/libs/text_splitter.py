import json
from typing import List, Literal
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)


def merge_small_documents(documents: List[Document], merge_max_length: int = 100):
    """
    合并较小的文档块

    参数:
        documents (List): 文档列表，每个元素是包含 pageContent 和 metadata 的字典。
        min_length (int): 要合并的最大文档块长度，默认为 100。

    返回:
        List: 更新后的文档列表。
    """
    i = 0
    while i < len(documents):
        doc = documents[i]
        page_content = doc.page_content
        if len(page_content) < merge_max_length:
            if i < len(documents) - 1:
                documents[i + 1].page_content = (
                    page_content + "\n" + documents[i + 1].page_content
                )
            documents.pop(i)
        else:
            i += 1
    return documents


def extract_headers(
    doc: Document,
    headers=[
        {"key": "header6", "value": "###### "},
        {"key": "header5", "value": "##### "},
        {"key": "header4", "value": "#### "},
        {"key": "header3", "value": "### "},
        {"key": "header2", "value": "## "},
        {"key": "header1", "value": "# "},
    ],
):
    """
    提取文档块中的 Markdown 标题，并保存到 metadata 中。

    参数:
        doc (dict): 包含 pageContent 和 metadata 的字典对象。

    返回:
        dict: 更新后的 doc 对象。
    """
    page_content = doc.page_content
    lines = page_content.split("\n")

    for header in headers:
        key = header["key"]
        value = header["value"]
        doc.metadata[key] = []

        for line in lines:
            new_line = line.strip()
            if new_line.startswith(value):
                doc.metadata[key].append(new_line)

        doc.metadata[key] = json.dumps(doc.metadata[key])
    return doc


def add_headers_to_documents(documents: List[Document]):
    """
    遍历文档列表，提取标题并继承前一个文档的标题（如缺失）。

    参数:
        documents (List): 文档列表，每个元素是包含 pageContent 和 metadata 的字典。

    返回:
        List: 更新后的文档列表。
    """
    headers = [
        {"key": "header6", "value": "###### "},
        {"key": "header5", "value": "##### "},
        {"key": "header4", "value": "#### "},
        {"key": "header3", "value": "### "},
        {"key": "header2", "value": "## "},
        {"key": "header1", "value": "# "},
    ]
    for i, doc in enumerate(documents):

        # 提取当前文档的标题
        doc = extract_headers(doc, headers)

        cur_start_header = None
        for header in headers:
            key = header["key"]
            if len(json.loads(doc.metadata.get(key))) > 0:
                cur_start_header = header

        # 如果当前文档没有对应层级的标题，则尝试从上一个文档继承
        for header in headers:
            key = header["key"]
            if cur_start_header == None or len(header["value"]) < len(
                cur_start_header["value"]
            ):
                cur_headers = json.loads(doc.metadata.get(key))
                if not cur_headers:
                    cur_headers = json.loads(documents[i - 1].metadata.get(key))
                    if i > 0 and cur_headers:
                        doc.metadata[key] = json.dumps([cur_headers[0]])
                        doc.page_content = f"{cur_headers[0]}\n{doc.page_content}"
    return documents


class CustomMarkdownTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(
        self, documents: List[Document], merge_max_length=100, add_headers=True
    ):
        # 1. 默认使用 MarkdownTextSplitter 切分
        text_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_result = text_splitter.split_documents(documents=documents)

        # 2. 再合并 chunkSize 小于 100 的文档块
        split_result = merge_small_documents(
            documents=split_result, merge_max_length=merge_max_length
        )

        # 3. 最后给每个文档块添加标题
        if add_headers == True:
            split_result = add_headers_to_documents(documents=split_result)

        return split_result


def TextSplitter(
    provider: Literal[
        "recursiveCharacter",
        "markdownText",
        "customMarkdownText",
    ] = "recursiveCharacter",
    **kwargs,
):
    if provider == "recursiveCharacter":
        return RecursiveCharacterTextSplitter(**kwargs)
    elif provider == "markdownText":
        return MarkdownTextSplitter(**kwargs)
    elif provider == "customMarkdownText":
        return CustomMarkdownTextSplitter(**kwargs)
    elif provider == "semanticChunker":
        return SemanticChunker(**kwargs)
    else:
        raise ValueError(f"Provider {provider} not supported.")
