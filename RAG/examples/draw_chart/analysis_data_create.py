import os
import json
from collections import defaultdict

# 目录路径
data_dir = "RAG/output/evaluate_data"

# 只处理 rag_evaluate_v7.0 开头的文件
files = [
    f
    for f in os.listdir(data_dir)
    if f.startswith("rag_evaluate_v7.0") and f.endswith(".json")
]

# 统计结构
stats = defaultdict(
    lambda: {
        "contextRecall": [],
        "contextRelevance": [],
        "faithfulness": [],
        "answerRelevance": [],
        "answerCorrectness": [],
    }
)

for file in files:
    # 解析 type
    type_str = file.replace("rag_evaluate_v7.0_", "").replace(".json", "")
    filepath = os.path.join(data_dir, file)
    with open(filepath, "r") as f:
        items = json.load(f)
    for item in items:
        answer = item.get("answer", "")
        # 特殊处理
        if "我无法根据现有信息回答这个问题" in answer:
            faithfulness = 0.8
            answerRelevance = 0.8
        else:
            faithfulness = item.get("faithfulness", {}).get("score", 0)
            answerRelevance = item.get("answerRelevance", {}).get("score", 0)
        stats[type_str]["contextRecall"].append(
            item.get("contextRecall", {}).get("score", 0)
        )
        stats[type_str]["contextRelevance"].append(
            item.get("contextRelevance", {}).get("score", 0)
        )
        stats[type_str]["faithfulness"].append(faithfulness)
        stats[type_str]["answerRelevance"].append(answerRelevance)
        stats[type_str]["answerCorrectness"].append(
            item.get("answerCorrectness", {}).get("score", 0)
        )

# 计算平均值并输出
result = []
for type_str, values in stats.items():
    count = len(values["contextRecall"])
    avg = lambda x: round(sum(x) / count, 2) if count > 0 else 0
    result.append(
        {
            "type": type_str,
            "score": {
                "contextRecall": avg(values["contextRecall"]),
                "contextRelevance": avg(values["contextRelevance"]),
                "faithfulness": avg(values["faithfulness"]),
                "answerRelevance": avg(values["answerRelevance"]),
                "answerCorrectness": avg(values["answerCorrectness"]),
            },
        }
    )

# 按 type 排序输出
result = sorted(result, key=lambda x: x["type"])
# 输出到文件
with open("RAG/output/evaluate_data/analysis_data.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
