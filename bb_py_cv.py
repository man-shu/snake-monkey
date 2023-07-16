"""
Script to create the boxplots comparing variables for the Bronze Back, Python, Pit Viper model presentations.

Author: Himanshu (himanshuaggarwal1997@gmail.com)
"""
# %%
# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
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

# update FR
CV["FR"] = CV["FR"] + CV["SS"]

# calculate Frequency of Vocalisations
for df in [BB, CV, PY]:
    df["FV"] = df["VO"] / df["TT"]

# calculate data summary
stats_dir = "stats"
os.makedirs(stats_dir, exist_ok=True)
for df in [BB, CV, PY]:
    df.describe().to_csv(os.path.join(stats_dir, f"{df.loc[0][0]}-stats.csv"))
# variables to plot
vars = {
    "PD": "Passing Distance (cm)",
    "PR": "Distance (cm)",
    "FR": "Frequency of Fear Response (1/s)",
    "BS": "Frequency of Bipedal Standing (1/s)",
    "GP": "Gaze Percentage (%)",
    "FV": "Frequency of Vocalisations (1/s)",
}


# %%
# Compare BB vs PY vs CV using Mann-Whitney U test.
remove_cols = ["subj", "snake model", "order", "sex", "VO", "SS"]
BB_ = BB.drop(columns=remove_cols, errors="ignore")
PY_ = PY.drop(columns=remove_cols, errors="ignore")
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
hatches = ["..", "X", "O"]

# %%
for var, label in vars.items():
    d = {"Bronze Back": BB[var], "Pit Viper": CV[var], "Python": PY[var]}
    df = pd.DataFrame(data=d)
    # plot
    fig = plt.figure()
    ax1 = plt.subplot2grid((15, 1), (1, 0), rowspan=14, ylabel=label)
    ax2 = plt.subplot2grid((15, 1), (0, 0))
    sns.boxplot(data=df, palette=colors, showfliers=False, ax=ax1)
    ax1 = insert_hatches(ax1, hatches)

    ax2 = insert_stats(ax2, p_bb_py[var][0], df, [0, 2], y_offset=6)
    ax2 = insert_stats(ax2, p_bb_cv[var][0], df, [0, 1])
    ax2 = insert_stats(ax2, p_py_cv[var][0], df, [1, 2])
    sns.despine(left=True, ax=ax1)
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
