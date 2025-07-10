import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_json("RAG/datas/eval_chart/file_parser_eval_data_1.json")

# 排除不需要的列
exclude_cols = [
    "Overall Edit EN",
    "Overall Edit ZH",
    "Methods",
]
metrics = [col for col in df.columns if col not in exclude_cols]

# 雷达图参数
labels = metrics
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, row in df.iterrows():
    values = [row[m] for m in metrics]
    values += values[:1]  # 闭合
    # 先画线，获取当前颜色
    (line,) = ax.plot(angles, values, label=row["Methods"], linewidth=2)
    ax.fill(angles, values, color=line.get_color(), alpha=0.1)
    # 在每个顶点加上与线条颜色一致的圆点
    ax.scatter(angles, values, s=50, zorder=3, color=line.get_color())

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_yticklabels([])  # 隐藏雷达图圆环上的数值
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.title("")
plt.tight_layout()
plt.show()
