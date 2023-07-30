import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns


def insert_hatches(ax, hatches):
    """
    Insert hatches into boxplots.
    """
    # select the correct patches
    patches = [
        patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch
    ]
    # iterate through the patches for each subplot
    for patch, hatch in zip(patches, hatches):
        patch.set_hatch(hatch)

    return ax


def insert_stats(ax, p_val, data, loc=[], h=2, y_offset=0, x_n=3):
    """
    Insert p-values from statistical tests into boxplots.
    """
    max_y = data.max().max()
    h = h / 100 * max_y
    y_offset = y_offset / 100 * max_y
    x1, x2 = loc[0], loc[1]
    if x1 == 0:
        x1 = 0.165 * (x_n - 1)
    if x2 == x_n - 1:
        x2 = 0.835 * (x_n - 1)
    y = max_y + h + y_offset
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c="0.25")
    if p_val < 0.0001:
        text = f"****"
    if p_val < 0.001:
        text = f"***"
    elif p_val < 0.01:
        text = f"**"
    elif p_val < 0.05:
        text = f"*"
    else:
        text = f"ns"
    ax.text(
        (x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", color="0.25"
    )
    ax.set_xticks([*range(0, x_n)])
    ax.axis("off")
    return ax


def plot_cv_indices(
    splits,
    X,
    y,
    group,
    out_dir,
    lw=10,
):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots()
    cmap_data = plt.cm.tab20
    cmap_cv = plt.cm.coolwarm
    if type(splits) is not list:
        splits = list(splits)
    n_splits = len(splits)
    _, y = np.unique(y, return_inverse=True)
    _, group = np.unique(group, return_inverse=True)
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(splits):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )
    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)),
        [ii + 1.5] * len(X),
        c=y,
        marker="_",
        lw=lw,
        cmap=cmap_data,
    )
    ax.scatter(
        range(len(X)),
        [ii + 2.5] * len(X),
        c=group,
        marker="_",
        lw=lw,
        cmap=cmap_data,
    )
    # Formatting
    yticklabels = [*range(n_splits)] + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
    )
    ax.set_title(f"Train/test splits")
    plot_file = f"cv_splits.png"
    plot_file = os.path.join(out_dir, plot_file)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)


def chance_level(df):
    classes = []
    n_splits = len(df)
    for _, row in df.iterrows():
        classes.extend(df["expected_labels"].tolist())
        classes.extend(df["predicted_labels"].tolist())
    classes = np.concatenate(classes)
    classes = np.asarray(classes)
    classes = np.unique(classes)
    df["labels"] = [classes for split in range(n_splits)]
    df["n_labels"] = [len(classes) for split in range(n_splits)]
    df["chance"] = [1 / len(classes) for split in range(n_splits)]
    return df


def plot_confusion(df, results_dir):
    # plot confusion matrices
    expected_labels = np.concatenate(df["expected_labels"].to_numpy())
    predicted_labels = np.concatenate(df["predicted_labels"].to_numpy())
    cm = confusion_matrix(expected_labels, predicted_labels, normalize="pred")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)
    tick_marks = np.arange(len(df["labels"].iloc[0]))
    tick_marks = tick_marks + 0.5
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(df["labels"].iloc[0])
    ax.set_yticklabels(df["labels"].iloc[0])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # ax.set_title(
    #     f"Predicting snake presented based on monkey gestural behavior"
    # )
    plot_file = os.path.join(
        results_dir,
        f"confusion.png",
    )
    plt.savefig(
        plot_file,
        bbox_inches="tight",
        dpi=600,
    )
    plot_file = os.path.join(
        results_dir,
        f"confusion.tiff",
    )
    plt.savefig(
        plot_file,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()
