"""
Script to create the boxplots comparing variables for the model, live and dead Pit Viper presentations.

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
from utils.hypo_tests import model_dead_v_live
import warnings

warnings.filterwarnings("ignore")

# %%
# read data
data_root = "data"
# only importing CV data
CV = pd.read_csv(os.path.join(data_root, "CV.csv"))
CV_live = pd.read_csv(os.path.join(data_root, "CV-live.csv"))
CV_dead = pd.read_csv(os.path.join(data_root, "CV-dead.csv"))

# update FR
CV["FR"] = CV["FG"] + CV["SS"]

# %%
# Compare model and dead vs live data using Mann-Whitney U test.
remove_cols = ["subj", "id", "snake model", "order", "sex", "VO", "SS", "TT"]
CV_ = CV.drop(columns=remove_cols, errors="ignore")
CV_dead_ = CV_dead.drop(columns=remove_cols, errors="ignore")
CV_live_ = CV_live.drop(columns=remove_cols, errors="ignore")
# get p-values
p_model_live, p_model_dead, p_dead_live = model_dead_v_live(
    CV_, CV_dead_, CV_live_
)
# %%
# create directory for figures
os.makedirs("plots", exist_ok=True)

# %%
# set plot style elements
sns.set(context="paper", font_scale=0.75)
sns.set_style("white")
plt.rcParams["figure.figsize"] = [1.5, 4.5]
# set up color palette
green_shades = sns.light_palette(sns.color_palette("colorblind")[2])
colors = [
    green_shades[-1],
    green_shades[2],
    green_shades[0],
]
# set hatch patterns, only first three being used here
hatches = ["XX", "//", "\\\\", "+", "x", "o", "O", ".", "*"]

# %%
# plot Proximity
CV_model_ = CV.copy()
CV_dead_ = CV_dead.copy()
n_obs = CV_model_.shape[0]
Samples = pd.DataFrame(np.empty((n_obs)) * np.nan)
Samples["Model"] = CV_model_["PR"]
Samples["Dead"] = CV_dead_["PR"]
Samples = Samples.drop(columns=[0])
fig = plt.figure()
ax1 = plt.subplot2grid(
    (45, 1), (1, 0), rowspan=44, ylabel="Closest Distance (cm)"
)
ax2 = plt.subplot2grid((45, 1), (0, 0))
sns.boxplot(data=Samples, showfliers=False, palette=colors, ax=ax1)
ax1 = insert_hatches(ax1, hatches)
ax2 = insert_stats(ax2, p_model_dead["PR"][0], Samples, [0, 1], x_n=2)
sns.despine(left=True, ax=ax1)
plt.ylabel("{}".format("Closest Distance (cm)"))
plt.savefig("plots/PR-model_dead-boxplot.tiff", bbox_inches="tight", dpi=600)
plt.savefig("plots/PR-model_dead-boxplot.jpg", bbox_inches="tight", dpi=600)
plt.close()

# %%
# plot rest of the variables
remove_cols = ["subj", "snake model", "sex", "order", "VO", "PR", "SS", "TT"]
CV_model_ = CV.copy().drop(columns=remove_cols, errors="ignore")
CV_model_ = CV_model_[CV_model_.PD < 500]
CV_live_ = CV_live.drop(columns=remove_cols, errors="ignore")
CV_dead_ = CV_dead.drop(columns=remove_cols, errors="ignore")
col_filename = list(CV_model_.columns)
rename_cols = {
    "BS": "Frequency of Bi-pedal Standing (1/s)",
    "GP": "Gaze Percentage (%)",
    "PD": "Passing Distance (cm)",
    "FR": "Frequency of Fear Response (1/s)",
}
CV_model_ = CV_model_.rename(columns=rename_cols)
CV_live_ = CV_live_.rename(columns=rename_cols)
CV_dead_ = CV_dead_.rename(columns=rename_cols)
p_dead_live = p_dead_live.rename(columns=rename_cols)
p_model_live = p_model_live.rename(columns=rename_cols)
p_model_dead = p_model_dead.rename(columns=rename_cols)
cols = list(CV_model_.columns)
n_obs = CV_model_.shape[0]
for col in cols:
    filename = cols.index(col)
    Samples = pd.DataFrame(np.empty((n_obs)) * np.nan)
    Samples["Model"] = CV_model_[col]
    Samples["Dead"] = CV_dead_[col]
    Samples["Live"] = CV_live_[col]
    Samples = Samples.drop(columns=[0])
    fig = plt.figure()
    ax1 = plt.subplot2grid((15, 1), (1, 0), rowspan=14, ylabel=f"{col}")
    ax2 = plt.subplot2grid((15, 1), (0, 0))
    sns.boxplot(data=Samples, showfliers=False, palette=colors, ax=ax1)
    ax1 = insert_hatches(ax1, hatches)
    # insert p-values
    ax2 = insert_stats(ax2, p_model_live[col][0], Samples, [0, 2], y_offset=6)
    ax2 = insert_stats(ax2, p_model_dead[col][0], Samples, [0, 1])
    sns.despine(left=True, ax=ax1)
    plt.savefig(
        "plots/{}-model_dead_live-boxplot.tiff".format(col_filename[filename]),
        bbox_inches="tight",
        dpi=600,
    )
    plt.savefig(
        "plots/{}-model_dead_live-boxplot.jpg".format(col_filename[filename]),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

# %%
