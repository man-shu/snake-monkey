import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(context="paper")
sns.set_style("white")


# Load the data
overall_results_df = pd.read_csv(
    "results/Gesture_overall.csv", index_col=False
)
# create a new column with train/test ratio in the format 0.4/0.6
overall_results_df["train/test"] = (
    ((100 - overall_results_df["test"]).astype(int)).astype(str)
    + " / "
    + (overall_results_df["test"]).astype(int).astype(str)
)
models = ["SVC", "LDA"]

# plot overall results as a heatmap
for model in models:
    df_ = overall_results_df.pivot(
        index="cv", columns="train/test", values=model
    )
    df_ = df_.fillna(0)
    df_ = df_.astype(float)
    # plot heatmap
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        df_,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Accuracy (%)"},
        vmin=61,
        vmax=67,
    )
    ax.invert_yaxis()

    if model == "LDA":
        # create a box around 100-20 cell
        plt.gca().add_patch(
            plt.Rectangle(
                (2, 2),
                1,
                1,
                fill=False,
                edgecolor="black",
                lw=2,
            )
        )
    plt.title(
        "Variation of accuracy with test size and\nnumber of cross-validation splits for "
        + model
    )
    plt.xlabel("Train / test ratio")
    plt.ylabel("Number of cross-validation splits")
    plt.savefig(
        f"plots/Gesture_{model}_overall.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.savefig(
        f"plots/Gesture_{model}_overall.tiff",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()
