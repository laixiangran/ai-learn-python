import json
import math
from RAG.libs.chat import Chat
from RAG.libs.evaluator import BatchEvaluator
from RAG.libs.file_loader import FileLoader
from RAG.libs.embedding import Embedding
from RAG.libs.text_splitter import (
    TextSplitter,
)
from RAG.libs.utils import (
    doc_length_distribution,
    save_json_data,
)
from RAG.libs.vector_store import VectorStore

base_path = "RAG/datas/split_data/07_chunkSize_chunkOverlap"
collection_name = "07_chunkSize_chunkOverlap_collection"
is_add_store = True

# 块大小
chunk_sizes = [128, 256, 384, 512, 640, 768, 896, 1024]

# 块重叠（占块大小的比例）
overlap_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def chunkSize_chunkOverlap_evaluate(qa_data):
    chat_model = Chat(model="qwen2.5:14b", provider="ollama")
    vector_store = VectorStore(
        collection_name=collection_name,
        embedding_function=Embedding(model="nomic-embed-text", provider="ollama"),
        provider="chroma",
    )
    result = []
    for chunk_size in chunk_sizes:
        for overlap_ratio in overlap_ratios:
            type = f"{chunk_size}_{overlap_ratio}"
            filter = {"type": type}
            output_path = f"RAG/datas/evaluate_data/rag_evaluate_v7.0_{type}.json"
            eval_result = BatchEvaluator(
                chat_model=chat_model,
                vector_store=vector_store,
                qa_data=qa_data,
                top_k=3,
                filter=filter,
                output_path=output_path,
            )
            print("-" * 20, type, "最终评分", "-" * 20)
            res = {"type": type, "score": eval_result["score"]}
            print(res)
            result.append(res)
    print("*" * 20, "最终评分", "*" * 20)
    print(result)


def chunkSize_chunkOverlap_chunker(
    documents=None,
    chunk_size=500,
    chunk_overlap=50,
    overlap_ratio=0.1,
):
    type = f"{chunk_size}_{overlap_ratio}"

    # 向量模型
    text_splitter = TextSplitter(
        provider="recursiveCharacter",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(documents=documents)
    print("-" * 20, type, "-" * 20)
    doc_length_distribution(documents)

    # 向量存储
    if is_add_store == True:
        embeddings = Embedding(model="nomic-embed-text", provider="ollama")
        vector_store = VectorStore(
            collection_name=collection_name,
            embedding_function=embeddings,
            provider="chroma",
        )
        ids = []
        for i, text in enumerate(documents):
            text.metadata["category"] = "少儿编程"
            text.metadata["type"] = type
            ids.append(type + str(i))
        vector_store.add_documents(documents=documents, ids=ids)

    # 保存到文件
    output_path = f"{base_path}/{type}.json"
    save_json_data(documents, output_path)

    return documents


def chunkSize_chunkOverlap_splitting():
    # 加载文件
    file_loader = FileLoader(
        file_path="RAG/datas/2024少儿编程教育行业发展趋势报告.md",
        provider="customMarkdownLoader",
    )
    documents = file_loader.load()

    for chunk_size in chunk_sizes:
        for overlap_ratio in overlap_ratios:
            chunk_overlap = math.ceil(chunk_size * overlap_ratio)
            chunkSize_chunkOverlap_chunker(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                overlap_ratio=overlap_ratio,
            )


if __name__ == "__main__":
    # 按不同 chunk_size 和 chunk_overlap 进行切分
    chunkSize_chunkOverlap_splitting()

    # 评估不同 chunk_size 和 chunk_overlap 的指标表现
    qa_test_path = "RAG/datas/qa_test_10.json"
    with open(qa_test_path, "r") as f:
        qa_data = json.load(f)
    chunkSize_chunkOverlap_evaluate(qa_data)
