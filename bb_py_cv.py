"""
Script to create the boxplots comparing variables for the Bronze Back, Python, Pit Viper model presentations.

Author: Himanshu (himanshuaggarwal1997@gmail.com)
"""
# %%
# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from utils.plot_utils import insert_hatches, insert_stats
from utils.hypo_tests import bb_py_cv
import warnings

warnings.filterwarnings("ignore")

# %%
# read data
data_root = "data"
BB = pd.read_csv(os.path.join(data_root, "BB.csv"))
CV = pd.read_csv(os.path.join(data_root, "CV.csv"))
PY = pd.read_csv(os.path.join(data_root, "PY.csv"))
# variables to plot
vars = {
    "PD": "Distance (cm)",
    "Pr": "Distance (cm)",
    "FG": "Frequency of Fear Grimace (1/s)",
    "SS": "Frequency of Self Scratching (1/s)",
    "BS": "Frequency of Bipedal Standing (1/s)",
    "GP": "Gaze Percentage (%)",
}


# %%
# Compare BB vs PY vs CV using Mann-Whitney U test.
remove_cols = ["ID", "Age-Group", "DD", "FR", "VO"]
BB_ = BB.drop(columns=remove_cols)
PY_ = PY.drop(columns=remove_cols)
CV_ = CV.drop(columns=remove_cols)
# get p-values
p_bb_py, p_bb_cv, p_py_cv = bb_py_cv(BB_, PY_, CV_)

# %%
# create directory for figures
plot_dir = "plots"
os.makedirs("plots", exist_ok=True)

# %%
sns.set(context="paper")
sns.set_style("white")

plt.rcParams["figure.figsize"] = [3, 8]
# set up color palette
colors = [
    sns.color_palette("colorblind")[3],
    sns.color_palette("colorblind")[2],
    sns.color_palette("colorblind")[8],
]
# set hatch patterns, only first three being used here
hatches = ["o", "xx", "..."]

# %%
for var, label in vars.items():
    d = {"Bronze Back": BB[var], "Pit Viper": CV[var], "Python": PY[var]}
    df = pd.DataFrame(data=d)

    # remove data points
    if var in ["Pr", "PD"]:
        df = df.drop(index=37)
    # plot
    ax = sns.boxplot(data=df, palette=colors, fliersize=0)
    ax = insert_hatches(ax, hatches)
    ax = insert_stats(ax, p_bb_py[var][0], df, [0, 2], y_offset=6)
    ax = insert_stats(ax, p_bb_cv[var][0], df, [0, 1])
    ax = insert_stats(ax, p_py_cv[var][0], df, [1, 2])
    sns.swarmplot(data=df, color=".25", ax=ax)
    sns.despine(left=True, ax=ax)
    plt.ylabel(label)
    plt.savefig(
        os.path.join(plot_dir, f"{var}-boxplot.tiff"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.savefig(
        os.path.join(plot_dir, f"{var}-boxplot.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

# %%
# test fVoc BB vs PY vs CV using Mann-Whitney U test.
import scipy.stats as stats

fVoc = pd.read_csv(os.path.join(data_root, "freq-voc.csv"))
fVoc.dropna(inplace=True)
p_bb_py = stats.mannwhitneyu(fVoc["BB"], fVoc["PY"], alternative="two-sided")[
    1
]
p_bb_cv = stats.mannwhitneyu(fVoc["BB"], fVoc["CV"], alternative="two-sided")[
    1
]
p_py_cv = stats.mannwhitneyu(fVoc["PY"], fVoc["CV"], alternative="two-sided")[
    1
]
print("p_bb_py:", p_bb_py)
print("p_bb_cv:", p_bb_cv)
print("p_py_cv:", p_py_cv)
fVoc = fVoc.rename(
    columns={"BB": "Bronze Back", "PY": "Python", "CV": "Pit Viper"}
)
# plot Vocalisation Frequency per interaction
ax = sns.boxplot(data=fVoc, palette=colors, fliersize=0)
ax = insert_hatches(ax, hatches)
ax = insert_stats(ax, p_bb_py, fVoc, [0, 2], y_offset=6)
ax = insert_stats(ax, p_bb_cv, fVoc, [0, 1])
ax = insert_stats(ax, p_py_cv, fVoc, [1, 2])
sns.swarmplot(data=fVoc, color=".25", ax=ax)
sns.despine(left=True, ax=ax)
plt.ylabel("Vocalisation Frequency per interaction")
plt.savefig(
    os.path.join(plot_dir, f"freq-voc-boxplot.tiff"),
    bbox_inches="tight",
    dpi=600,
)
plt.savefig(
    os.path.join(plot_dir, f"freq-voc-boxplot.jpg"),
    bbox_inches="tight",
    dpi=600,
)
plt.close()
