from RAG.libs.file_loader import FileLoader


def langchain_markdown_loader_test():
    #  创建一个解析器对象
    file_loader = FileLoader(
        file_path="RAG/datas/2024少儿编程教育行业发展趋势报告.md",
        provider="customMarkdownLoader",
    )
    res = file_loader.load()
    print(res)


if __name__ == "__main__":
    langchain_markdown_loader_test()
