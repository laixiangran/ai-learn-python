from RAG.libs.text_splitter import TextSplitter


def text_splitter_test():
    text_splitter = TextSplitter("recursiveCharacterTextSplitter")
    print(text_splitter)


if __name__ == "__main__":
    text_splitter_test()
