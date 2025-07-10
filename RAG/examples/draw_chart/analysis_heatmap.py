import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取数据
with open("RAG/datas/eval_chart/analysis_data.json", "r") as f:
    data = json.load(f)

# 解析数据
chunk_sizes = sorted(list(set(int(item["type"].split("_")[0]) for item in data)))
overlap_ratios = sorted(list(set(float(item["type"].split("_")[1]) for item in data)))
metrics = list(data[0]["score"].keys())

# 组织数据为字典
results = {}
for metric in metrics:
    results[metric] = np.zeros((len(chunk_sizes), len(overlap_ratios)))
for item in data:
    cs = int(item["type"].split("_")[0])
    oratio = float(item["type"].split("_")[1])
    i = chunk_sizes.index(cs)
    j = overlap_ratios.index(oratio)
    for metric in metrics:
        results[metric][i, j] = item["score"][metric]

# 1. 折线图：每个 chunkSize 一条线，横轴为 overlap_ratio，纵轴为指标
# for metric in metrics:
#     plt.figure(figsize=(8, 5))
#     for i, cs in enumerate(chunk_sizes):
#         plt.plot(
#             overlap_ratios, results[metric][i], marker="o", label=f"chunkSize={cs}"
#         )
#     plt.xlabel("chunkOverlap Ratio")
#     plt.ylabel(metric)
#     plt.title(f"{metric} vs chunkOverlap Ratio")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# 2. 热力图：chunkSize 为 y 轴，overlap_ratio 为 x 轴，颜色为指标值
for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        results[metric],
        annot=True,
        xticklabels=overlap_ratios,
        yticklabels=chunk_sizes,
        cmap="YlGnBu",
    )
    plt.xlabel("chunkOverlap Ratio")
    plt.ylabel("chunkSize")
    plt.title(f"Heatmap of {metric}")
    plt.tight_layout()
    plt.show()
