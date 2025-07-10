import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 读取数据
with open("RAG/datas/eval_chart/analysis_data.json", "r") as f:
    data = json.load(f)

chunk_sizes = []
overlap_ratios = []
contextRecall = []
contextRelevance = []
faithfulness = []
answerRelevance = []
answerCorrectness = []

for item in data:
    cs, oratio = item["type"].split("_")
    chunk_sizes.append(int(cs))
    overlap_ratios.append(float(oratio))
    contextRecall.append(item["score"]["contextRecall"])
    contextRelevance.append(item["score"]["contextRelevance"])
    faithfulness.append(item["score"]["faithfulness"])
    answerRelevance.append(item["score"]["answerRelevance"])
    answerCorrectness.append(item["score"]["answerCorrectness"])

# 找到每个指标得分最高的参数组合
max_contextRecall_idx = contextRecall.index(max(contextRecall))
max_contextRelevance_idx = contextRelevance.index(max(contextRelevance))
max_faithfulness_idx = faithfulness.index(max(faithfulness))
max_answerRelevance_idx = answerRelevance.index(max(answerRelevance))
max_answerCorrectness_idx = answerCorrectness.index(max(answerCorrectness))

print(
    "contextRecall最高：chunkSize={}, chunkOverlap Ratio={}, 得分={}".format(
        chunk_sizes[max_contextRecall_idx],
        overlap_ratios[max_contextRecall_idx],
        contextRecall[max_contextRecall_idx],
    )
)
print(
    "contextRelevance最高：chunkSize={}, chunkOverlap Ratio={}, 得分={}".format(
        chunk_sizes[max_contextRelevance_idx],
        overlap_ratios[max_contextRelevance_idx],
        contextRelevance[max_contextRelevance_idx],
    )
)
print(
    "faithfulness最高：chunkSize={}, chunkOverlap Ratio={}, 得分={}".format(
        chunk_sizes[max_faithfulness_idx],
        overlap_ratios[max_faithfulness_idx],
        faithfulness[max_faithfulness_idx],
    )
)
print(
    "answerRelevance最高：chunkSize={}, chunkOverlap Ratio={}, 得分={}".format(
        chunk_sizes[max_answerRelevance_idx],
        overlap_ratios[max_answerRelevance_idx],
        answerRelevance[max_answerRelevance_idx],
    )
)
print(
    "answerCorrectness最高：chunkSize={}, chunkOverlap Ratio={}, 得分={}".format(
        chunk_sizes[max_answerCorrectness_idx],
        overlap_ratios[max_answerCorrectness_idx],
        answerCorrectness[max_answerCorrectness_idx],
    )
)

# 绘制三维气泡图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# contextRecall
ax.scatter(
    chunk_sizes,
    overlap_ratios,
    contextRecall,
    s=[v * 300 for v in contextRecall],
    c="r",
    alpha=0.6,
    label="contextRecall",
)
# contextRelevance
ax.scatter(
    chunk_sizes,
    overlap_ratios,
    contextRelevance,
    s=[v * 300 for v in contextRelevance],
    c="g",
    alpha=0.6,
    label="contextRelevance",
)
# faithfulness
ax.scatter(
    chunk_sizes,
    overlap_ratios,
    faithfulness,
    s=[v * 300 for v in faithfulness],
    c="y",
    alpha=0.6,
    label="faithfulness",
)
# answerRelevance
ax.scatter(
    chunk_sizes,
    overlap_ratios,
    answerRelevance,
    s=[v * 300 for v in answerRelevance],
    c="m",
    alpha=0.6,
    label="answerRelevance",
)
# answerCorrectness
ax.scatter(
    chunk_sizes,
    overlap_ratios,
    answerCorrectness,
    s=[v * 300 for v in answerCorrectness],
    c="b",
    alpha=0.6,
    label="answerCorrectness",
)

ax.set_xlabel("chunkSize")
ax.set_ylabel("chunkOverlap Ratio")
ax.set_zlabel("Score")
ax.set_title("各指标 与 chunkSize、chunkOverlap 的关系")
ax.legend()
plt.tight_layout()
plt.show()
