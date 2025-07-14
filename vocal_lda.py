import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    ellipse.set_alpha(0.2)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    # load data
    snakeCalls = pd.read_csv("data/Vocal.csv")
    # drop non variables
    X = snakeCalls.drop(columns=["id", "snake_type", "subj"])
    print(X)
    # standardize
    X = StandardScaler().fit_transform(X)
    print(X)
    # summarize
    print(snakeCalls["id"].value_counts())

    # create a dictionary to map the target names to the target ids
    # just for plotting
    target_dict = pd.Series(pd.unique(snakeCalls["id"])).to_dict()
    print(target_dict)
    target_names = target_dict.values()
    target_ids = target_dict.keys()

    # labels for LDA
    y = snakeCalls["id"]
    print(y)

    # start LDA
    lda = LinearDiscriminantAnalysis()
    X_r = lda.fit(X, y).transform(X)

    sns.set(context="talk")
    sns.set_style("white")

    # plot with confidence ellipses
    plt.rcParams["figure.figsize"] = [8, 8]
    # select the colors
    # keh = green
    # ker = grey
    # k-krah = yellow
    # kia = red
    colors = [
        sns.color_palette("colorblind")[8],
        sns.color_palette("colorblind")[2],
        sns.color_palette("colorblind")[7],
        sns.color_palette("colorblind")[1],
    ]

    count = 0
    for color, i, class_ in zip(colors, lda.classes_, lda.classes_):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=class_)
        # plt.scatter(mean_Scaled[count, 0], mean_Scaled[count, 1], color=color, marker='x')
        ax = plt.gca()
        confidence_ellipse(
            X_r[y == i, 0], X_r[y == i, 1], ax, n_std=2.0, facecolor=color
        )
        confidence_ellipse(
            X_r[y == i, 0], X_r[y == i, 1], ax, n_std=1.0, facecolor=color
        )
        count += 1
        plt.xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.2f}%)")
        plt.ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.2f}%)")
    plt.legend(loc="best", shadow=False, scatterpoints=1, title="Call ID")
    sns.despine(top=True, right=True, ax=ax)
    plot_file = os.path.join("plots", "Vocal_LDA")
    plt.savefig(f"{plot_file}.png", bbox_inches="tight", dpi=600)
    plt.savefig(f"{plot_file}.tiff", bbox_inches="tight", dpi=300)
    plt.close()
