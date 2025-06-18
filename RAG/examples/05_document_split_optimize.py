from RAG.libs.file_loader import FileLoader
from RAG.libs.text_splitter import TextSplitter


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
        provider="recursiveCharacter",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_result = text_splitter.split_documents(file_loader_result)
    print(len(split_result))

    # 2. 基于 MarkdownTextSplitter 进行文件切分
    text_splitter = TextSplitter(
        provider="markdownText",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_result = text_splitter.split_documents(file_loader_result)
    print(len(split_result))

    # 3. 基于 MarkdownTextSplitter 进行文件切分后，再合并 chunkSize 小于 100（通过 merge_max_length 设置）的文档块并添加标题
    text_splitter = TextSplitter(
        provider="customMarkdownText",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_result = text_splitter.split_documents(split_result, merge_max_length=100)
    print(len(split_result))
    print(split_result)


if __name__ == "__main__":
    demo()
