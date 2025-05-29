# %%
# import libraries
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from utils.plot_utils import (
    plot_cv_indices,
    chance_level,
    plot_confusion,
)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import umap

# %%
# read data
data_root = "data"
BB = pd.read_csv(os.path.join(data_root, "BB.csv"))
CV = pd.read_csv(os.path.join(data_root, "CV.csv"))
PY = pd.read_csv(os.path.join(data_root, "PY.csv"))

# concatenate data
df = pd.concat([BB, CV, PY], ignore_index=True)

# %%
# calculate feature
df["FR"] = df["FG"] + df["SS"]
# get classes and groups
df["snake model"] = np.select(
    [
        df["snake model"] == "BB",
        df["snake model"] == "CV",
        df["snake model"] == "PY",
    ],
    ["Bronze Back", "Pit Viper", "Python"],
    default="0",
)
classes = df["snake model"].to_numpy(dtype=object)
groups = df["subj"].to_numpy(dtype=object)
# drop unnecessary columns
df = df.drop(
    columns=["order", "sex", "VO", "TT", "FG", "SS", "snake model", "subj"]
)
# convert to numpy array
X = df.to_numpy()


# %%
# model def and apply
scaler = StandardScaler()
reducer = umap.UMAP(random_state=1234)
scale_red = make_pipeline(scaler, reducer)

embedding = scale_red.fit_transform(X)
embedding.shape

# %%
# plotting
sns.set(context="paper")
sns.set_style("white")

# do a scatter plot with labels
# Get unique classes
unique_classes = np.unique(classes)
# Create color mapping
class_to_color = {"Bronze Back": 3, "Pit Viper": 2, "Python": 8}

# Plot each class separately
for clas in unique_classes:
    mask = classes == clas
    plt.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c=[sns.color_palette("colorblind")[class_to_color[clas]]],
        label=clas,
    )

plt.title("UMAP projection of the Snake dataset")
plt.legend()
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig("plots/gesture_umap.png", dpi=600, bbox_inches="tight")
plt.savefig("plots/gesture_umap.tiff", dpi=600, bbox_inches="tight")
plt.close()
# %%
