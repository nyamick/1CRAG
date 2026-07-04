# -*- coding: utf-8 -*-
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("/home/claude/project/results/metrics.json", encoding="utf-8") as f:
    results = json.load(f)

methods = [r["method"] for r in results]
ks = [1, 3, 5]
x = np.arange(len(ks))
width = 0.25

fig, ax = plt.subplots(figsize=(6.5, 4))
for i, r in enumerate(results):
    vals = [r[f"Recall@{k}"] for k in ks]
    ax.bar(x + i * width, vals, width, label=r["method"])

ax.set_xticks(x + width)
ax.set_xticklabels([f"Recall@{k}" for k in ks])
ax.set_ylabel("Значение метрики")
ax.set_title("Сравнение методов retrieval по Recall@k")
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("/home/claude/project/results/recall_comparison.png", dpi=150)
print("saved plot")
