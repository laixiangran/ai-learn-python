import json
from RAG.libs.chat import Chat
from RAG.libs.embedding import Embedding
from RAG.libs.evaluator import BatchEvaluator
from RAG.libs.vector_store import VectorStore


def demo():
    with open("RAG/datas/evaluate_data/qa_test_1.json", "r") as f:
        qa_data = json.load(f)
    chat_model = Chat(model="qwen2.5:14b", provider="ollama")
    embedding = Embedding(model="nomic-embed-text", provider="ollama")
    vector_store = VectorStore(
        collection_name="01_simple_rag_collection",
        embedding_function=embedding,
        provider="chroma",
    )
    output_path = "RAG/datas/evaluate_data/rag_evaluate_v1.0.json"
    eval_result = BatchEvaluator(
        chat_model=chat_model,
        vector_store=vector_store,
        qa_data=qa_data,
        top_k=3,
        output_path=output_path,
    )
    print("-" * 20, "最终评分", "-" * 20)
    print(eval_result["evaluate_result"])


if __name__ == "__main__":
    demo()
