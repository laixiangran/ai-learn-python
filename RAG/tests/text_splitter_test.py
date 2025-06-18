from langchain_core.documents import Document
from RAG.libs.text_splitter import (
    TextSplitter,
    merge_small_documents,
    add_headers_to_documents,
)


def text_splitter_test(ducuments):
    text_splitter = TextSplitter("recursiveCharacter")
    split_result = text_splitter.split_documents(ducuments)
    print(split_result)


def custom_markdown_text_splitter_test(ducuments):
    text_splitter = TextSplitter("customMarkdownText")
    split_result = text_splitter.split_documents(ducuments)
    print(split_result)


def text_handler(ducuments):
    split_result = merge_small_documents(ducuments)
    for text in split_result:
        print(text.page_content)
        print("**" * 40)

    split_result = add_headers_to_documents(split_result)
    print(split_result)


if __name__ == "__main__":
    ducuments = [
        Document(
            page_content="# hello world11\n## hello world22\n### hello world22\ndsadsadasd\n## hello world33\nhhshdsah",
            metadata={"category": "少儿编程"},
        ),
        Document(
            page_content="## hello world22",
            metadata={"category": "少儿编程"},
        ),
        Document(
            page_content="### hello world33 sadsadsadsad\n### hello world5555",
            metadata={"category": "少儿编程"},
        ),
        Document(
            page_content="hello world00000000",
            metadata={"category": "少儿编程"},
        ),
        Document(
            page_content="## hello world66666\n### hello world777777",
            metadata={"category": "少儿编程"},
        ),
    ]

    text_splitter_test(ducuments)
    custom_markdown_text_splitter_test(ducuments)
    text_handler(ducuments)
