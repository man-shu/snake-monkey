import pandas as pd
import os
from utils.plot_utils import (
    plot_cv_indices,
    chance_level,
    plot_confusion,
)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data_root = "data"
prim_det = pd.read_csv(os.path.join(data_root, "primary_detector.csv"))
ranks = pd.read_csv(os.path.join(data_root, "Social rank.csv"))

# merge data
df = pd.merge(prim_det, ranks, on="subj", how="right")
df = df.sort_values(by=["rank"]).reset_index(drop=True)
df.to_csv("data/ranks_snakes_detected.csv", index=False)

data = {}

for row in df.iterrows():
    row = row[1]
    if row["subj"] not in data:
        data[row["subj"]] = {
            "rank": row["rank"],
            "CV": 0,
            "BB": 0,
            "PY": 0,
        }
    if row["snake"] in ["CV", "BB", "PY"]:
        data[row["subj"]][row["snake"]] += 1
    else:
        continue

df = pd.DataFrame(data).T
df.to_csv("data/ranks_snakes_detected_counts.csv")

# divide by total detections in each column
for snake in ["CV", "BB", "PY"]:
    df[snake] = df[snake] / df[snake].sum()
df.to_csv("data/ranks_snakes_detected_probabilities.csv")

# plot
fig, ax = plt.subplots(figsize=(10, 5))
df.plot(kind="bar", x="rank", y=["CV", "BB", "PY"], stacked=True, ax=ax)
plt.xlabel("Social rank")
plt.ylabel("Detection probability")
plt.legend(title="Snake")
plt.title("Detection probability by social rank")
plt.savefig("plots/detection_probability.png")
