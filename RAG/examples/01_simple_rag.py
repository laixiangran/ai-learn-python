from RAG.libs.file_loader import FileLoader
from RAG.libs.text_splitter import TextSplitter
from RAG.libs.embedding import Embedding
from RAG.libs.vector_store import VectorStore
from RAG.libs.chat import Chat
from RAG.libs.prompts import Prompts


def demo():
    # 加载文件
    file_loader = FileLoader(
        file_path="RAG/datas/2024少儿编程教育行业发展趋势报告.md",
        provider="customMarkdownLoader",
    )
    file_loader_result = file_loader.load()

    # 文件切分
    text_splitter = TextSplitter(
        provider="recursiveCharacterTextSplitter", chunk_size=500, chunk_overlap=50
    )
    split_result = text_splitter.split_documents(file_loader_result)

    # 向量存储
    embedding = Embedding(model="nomic-embed-text", provider="ollama")
    vector_store = VectorStore(
        collection_name="01_simple_rag_collection",
        embedding_function=embedding,
        provider="chroma",
    )
    ids = []
    for i, text in enumerate(split_result):
        text.metadata["category"] = "少儿编程"
        ids.append(str(i))
    vector_store.add_documents(documents=split_result, ids=ids)

    # 向量检索
    search_result = vector_store.similarity_search_with_score(
        query="什么是少儿编程教育？", k=3
    )

    # 根据检索结果和问题构造提示词
    context = []
    for result in search_result:
        context.append(result[0].page_content)
    context = "\n".join(context)
    question = "什么是少儿编程教育？"
    input = Prompts.rag_answer_prompt.format(context=context, question=question)
    print("input：", input)

    # llm 回答
    chat_model = Chat(model="qwen2.5:14b", provider="ollama")
    chat_result = chat_model.invoke(input=input)
    print("answer：", chat_result.content)


if __name__ == "__main__":
    demo()
