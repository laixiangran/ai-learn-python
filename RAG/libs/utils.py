import json
from langchain_core.documents import Document
import pandas as pd


def read_json_data(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


def save_json_data(json_data, output_path):
    if type(json_data[0]) == Document:
        newData = []
        for doc in json_data:
            newData.append({"page_content": doc.page_content, "metadata": doc.metadata})
        json_data = newData
    json_data = json.dumps(json_data, indent=2, ensure_ascii=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_data)


def json_data_to_document(json_data):
    documens = []
    for doc in json_data:
        newDoc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documens.append(newDoc)
    return documens


def doc_length_distribution(documens):
    """
    统计文档长度的分布情况。

    该函数接受一个包含文档数据的列表作为输入，计算并返回文档长度（以字符数衡量）的各种统计量，
    包括计数、平均长度、标准差、最小值、最大值以及特定百分位数。

    参数:
    json_data (list): 包含文档数据的列表。每个元素是一个字典或Document对象，包含文档的页面内容和元数据。

    返回:
    pd.Series: 包含文档长度分布统计信息的Pandas Series对象。
    """
    if type(documens[0]) == Document:
        newData = []
        for doc in documens:
            newData.append({"page_content": doc.page_content, "metadata": doc.metadata})
        documens = newData
    print(f"docs count: {len(documens)}")
    print("doc length distribution:")
    result = pd.Series([len(d["page_content"]) for d in documens]).describe(
        [0.25, 0.5, 0.75, 0.9, 0.97, 0.99]
    )
    print(result)
    return result
