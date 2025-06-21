import json
import re
from cv2 import merge
from langchain_experimental.text_splitter import SemanticChunker
from RAG.libs.chat import Chat
from RAG.libs.evaluator import BatchEvaluator
from RAG.libs.file_loader import FileLoader
from RAG.libs.embedding import Embedding
from RAG.libs.prompts import Prompts
from RAG.libs.text_splitter import (
    TextSplitter,
    merge_small_documents,
    add_headers_to_documents,
)
from RAG.libs.utils import (
    doc_length_distribution,
    save_json_data,
)
from RAG.libs.vector_store import VectorStore

base_path = "RAG/datas/split_data/semantic_splitting"
collection_name = "06_semantic_splitting_collection"
is_add_store = True
breakpoint_threshold_types = [
    "percentile",
    "percentile-300",
    "standard_deviation",
    "interquartile",
    "gradient",
]


def semantic_splitting_evaluate(qa_data):
    chat_model = Chat(model="qwen2.5:14b", provider="ollama")
    vector_store = VectorStore(
        collection_name=collection_name,
        embedding_function=Embedding(model="nomic-embed-text", provider="ollama"),
        provider="chroma",
    )
    result = []
    for i, type in enumerate(breakpoint_threshold_types):
        filter = {"type": type}
        output_path = f"RAG/datas/evaluate_data/rag_evaluate_v6.0_{type}.json"
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


def semantic_chunker(
    documents=None,
    type="percentile",
):
    # 向量模型
    embeddings = Embedding(model="nomic-embed-text", provider="ollama")

    # 使用 Embedding 模型进行切分
    # sentence_split_regex：默认按英文的句号、问号、感叹号切分。切分中文文档时，要替换成按中文的句号、问号、感叹号和换行符切分
    number_of_chunks = None
    new_type = type
    if len(type.split("-")) > 1:
        new_type = type.split("-")[0]
        number_of_chunks = int(type.split("-")[1])
    print("number_of_chunks", number_of_chunks)
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=new_type,
        sentence_split_regex=r"(?<=[。！？\n])",
        number_of_chunks=number_of_chunks,
    )
    documents = text_splitter.split_documents(documents=documents)
    print("-" * 20, type, "-" * 20)
    doc_length_distribution(documents)

    # 将较大的文档进行二次切分
    text_splitter = TextSplitter(
        provider="recursiveCharacter",
        chunk_size=500,
        chunk_overlap=0,
    )
    documents = text_splitter.split_documents(documents)
    doc_length_distribution(documents)

    # 合并较小的块和添加标题
    documents = merge_small_documents(documents, merge_max_length=100)
    documents = add_headers_to_documents(documents)
    doc_length_distribution(documents)

    # 向量存储
    if is_add_store == True:
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


def semantic_splitting():
    # 加载文件
    file_loader = FileLoader(
        file_path="RAG/datas/2024少儿编程教育行业发展趋势报告.md",
        provider="customMarkdownLoader",
    )
    documents = file_loader.load()

    for type in breakpoint_threshold_types:
        semantic_chunker(
            documents=documents,
            type=type,
        )


if __name__ == "__main__":
    qa_test_path = "RAG/datas/evaluate_data/qa_test_10.json"
    with open(qa_test_path, "r") as f:
        qa_data = json.load(f)

    # 语义切分
    semantic_splitting()

    # 评估语义切分效果
    semantic_splitting_evaluate(qa_data)
