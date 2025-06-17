from langchain_core.documents import Document
from RAG.libs.embedding import Embedding
from RAG.libs.vector_store import VectorStore


def chroma_verctor_store_test():
    embedding = Embedding(model="nomic-embed-text", provider="ollama")
    vector_store = VectorStore(
        collection_name="my_collection", embedding_function=embedding, provider="chroma"
    )
    document_1 = Document(
        page_content="什么是少儿编程教育？", metadata={"category": "少儿编程"}
    )
    document_2 = Document(
        page_content="国家政策对少儿编程教育的影响体现在哪些方面？",
        metadata={"category": "少儿编程"},
    )
    documents = [document_1, document_2]
    ids = ["1", "2"]
    res = vector_store.add_documents(documents=documents, ids=ids)
    print(res)
    results = vector_store.similarity_search(query="少儿编程", k=1)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")


if __name__ == "__main__":
    chroma_verctor_store_test()
