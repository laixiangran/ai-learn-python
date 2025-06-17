import re
from RAG.libs.file_loader import FileLoader
from RAG.libs.text_splitter import TextSplitter


def merge_small_documents(documents):
    i = 0
    while i < len(documents):
        doc = documents[i]
        page_content = doc.page_content
        # 判断 pageContent 是否小于 100 字符
        if len(page_content) < 100:
            # 如果不是最后一个元素，则合并到下一个
            if i < len(documents) - 1:
                documents[i + 1].page_content = (
                    page_content + "\n" + documents[i + 1].page_content
                )
            # 删除当前元素
            documents.pop(i)
            i -= 1  # 回退索引
        else:
            i += 1  # 继续下一个元素
    return documents


def extract_headers(doc):
    """
    提取文档块中的 Markdown 标题，并保存到 metadata 中。

    参数:
        doc (dict): 包含 pageContent 和 metadata 的字典对象。

    返回:
        dict: 更新后的 doc 对象。
    """
    page_content = doc.page_content
    lines = page_content.split("\n")
    headers = [
        {"key": "header5", "value": "##### "},
        {"key": "header4", "value": "#### "},
        {"key": "header3", "value": "### "},
        {"key": "header2", "value": "## "},
        {"key": "header1", "value": "# "},
    ]

    for header in headers:
        key = header["key"]
        value = header["value"]
        doc.metadata[key] = []

        # 提取当前层级的标题
        for line in lines:
            if line.startswith(value):
                clean_title = re.sub(f"^{re.escape(value)}", "", line).strip()
                doc.metadata[key].append(clean_title)

    return doc


def add_headers_to_documents(documents):
    """
    遍历文档列表，提取标题并继承前一个文档的标题（如缺失）。

    参数:
        documents (list): 文档列表，每个元素是包含 pageContent 和 metadata 的字典。

    返回:
        list: 更新后的文档列表。
    """
    for i, doc in enumerate(documents):
        doc = extract_headers(doc)  # 提取当前文档的标题
        print(doc)
        metadata = doc.metadata
        # 如果当前文档没有 header1~header5，尝试从上一个文档继承
        for header_level in ["header1", "header2", "header3", "header4", "header5"]:
            if not metadata.get(header_level):
                if i > 0 and documents[i - 1].metadata.get(header_level):
                    pre_header = documents[i - 1].metadata[header_level][0]
                    doc.page_content = f"{pre_header}\n\n{doc.page_content}"
                    doc.metadata[header_level] = [pre_header]
    return documents


def demo():
    # 加载文件
    file_loader = FileLoader(
        file_path="RAG/datas/2024少儿编程教育行业发展趋势报告.md",
        provider="customMarkdownLoader",
    )
    file_loader_result = file_loader.load()

    # 基于特定字符的切分方法

    chunk_size = 500
    chunk_overlap = 50

    # 1. 基于 RecursiveCharacterTextSplitter 进行文件切分
    text_splitter = TextSplitter(
        provider="recursiveCharacterTextSplitter",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_result = text_splitter.split_documents(file_loader_result)
    for text in split_result:
        print(text.page_content)
        print("**" * 40)

    # 2. 基于 MarkdownTextSplitter 进行文件切分
    text_splitter = TextSplitter(
        provider="markdownTextSplitter",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_result = text_splitter.split_documents(file_loader_result)
    for text in split_result:
        print(text.page_content)
        print("**" * 40)

    # 3. 在 2 的基础上合并 chunkSize 小于 100 的文档块
    split_result = merge_small_documents(split_result)
    for text in split_result:
        print(text.page_content)
        print("**" * 40)

    # 在 3 的基础上给每个文档块添加标题
    split_result = add_headers_to_documents(split_result)
    for text in split_result:
        print(text.page_content)
        print("**" * 40)


if __name__ == "__main__":
    demo()
