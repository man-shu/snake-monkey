from scipy import stats
import pandas as pd
import os

TEST_ROOT = "tests"
os.makedirs(TEST_ROOT, exist_ok=True)


def model_dead_v_live(model, dead, live):
    """
    Compare model vs dead vs live data using Mann-Whitney U test.
    """
    d = {}
    all_dfs = []
    cols = list(model.columns)
    assert cols == list(dead.columns) == list(live.columns)
    for comparison in ["modelvdead", "modelvlive", "deadvlive"]:
        for col in cols:
            if comparison == "modelvdead":
                d[col] = stats.mannwhitneyu(
                    model[col], dead[col], alternative="two-sided"
                )[1]
            elif comparison == "modelvlive":
                if col == "PR":
                    continue
                else:
                    d[col] = stats.mannwhitneyu(
                        model[col], live[col], alternative="two-sided"
                    )[1]
            elif comparison == "deadvlive":
                if col == "PR":
                    continue
                else:
                    d[col] = stats.mannwhitneyu(
                        dead[col], live[col], alternative="two-sided"
                    )[1]
        df = pd.DataFrame(d, index=[0])
        df.to_csv(os.path.join(TEST_ROOT, f"{comparison}.csv"))
        all_dfs.append(df)

    return all_dfs[0], all_dfs[1], all_dfs[2]


def bb_py_cv(bb, py, cv):
    """
    Compare BB vs PY vs CV data using Mann-Whitney U test.
    """
    d = {}
    all_dfs = []
    cols = list(bb.columns)
    assert cols == list(py.columns) == list(cv.columns)
    for comparison in ["bbvpy", "bbvcv", "pyvcv"]:
        for col in cols:
            bb_col = bb[col].dropna()
            py_col = py[col].dropna()
            cv_col = cv[col].dropna()
            if comparison == "bbvpy":
                d[col] = stats.mannwhitneyu(
                    bb_col, py_col, alternative="two-sided"
                )[1]
            elif comparison == "bbvcv":
                d[col] = stats.mannwhitneyu(
                    bb_col, cv_col, alternative="two-sided"
                )[1]
            elif comparison == "pyvcv":
                d[col] = stats.mannwhitneyu(
                    py_col, cv_col, alternative="two-sided"
                )[1]
        df = pd.DataFrame(d, index=[0])
        df.to_csv(os.path.join(TEST_ROOT, f"{comparison}.csv"))
        all_dfs.append(df)

    return all_dfs[0], all_dfs[1], all_dfs[2]
