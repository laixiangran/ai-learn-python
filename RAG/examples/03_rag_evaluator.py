import json
from RAG.libs.chat import Chat
from RAG.libs.embedding import Embedding
from RAG.libs.evaluator import BatchEvaluator
from RAG.libs.vector_store import VectorStore


def demo():
    qa_test_path = "RAG/datas/evaluate_data/qa_test_10.json"
    with open(qa_test_path, "r") as f:
        qa_data = json.load(f)
    chat_model_name = "qwen2.5:14b"
    chat_model_provider = "ollama"
    embedding_model_name = "nomic-embed-text"
    embedding_model_provider = "ollama"
    chat_model = Chat(model=chat_model_name, provider=chat_model_provider)
    embedding = Embedding(model=embedding_model_name, provider=embedding_model_provider)
    top_k = 3
    collection_name = "01_simple_rag_collection"
    vector_store_provider = "chroma"
    vector_store = VectorStore(
        collection_name=collection_name,
        embedding_function=embedding,
        provider=vector_store_provider,
    )
    filter = None
    output_path = f"RAG/datas/evaluate_data/rag_evaluate_v1.0.json"
    eval_result = BatchEvaluator(
        chat_model=chat_model,
        vector_store=vector_store,
        qa_data=qa_data,
        top_k=top_k,
        filter=filter,
        output_path=output_path,
    )
    print(type, "-" * 20, "最终评分", "-" * 20)
    print(eval_result["evaluate_result"])


if __name__ == "__main__":
    demo()
