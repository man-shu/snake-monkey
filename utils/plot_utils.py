import matplotlib as mpl


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
