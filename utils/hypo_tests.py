from scipy import stats
import pandas as pd
import os

TEST_ROOT = "tests"
os.makedirs(TEST_ROOT, exist_ok=True)


def model_dead_v_live(model, dead, live):
    """
    Compare model and dead vs live data using Mann-Whitney U test.
    """
    d = {}
    # get intersection of columns in all dataframes
    modelvlive = set(model.columns).intersection(set(live.columns))
    deadvlive = set(dead.columns).intersection(set(live.columns))
    for col in modelvlive:
        d[col] = stats.mannwhitneyu(model[col], live[col])[1]
    df_modelvlive = pd.DataFrame(d, index=[0])
    df_modelvlive.to_csv(os.path.join(TEST_ROOT, "modelvlive.csv"))
    d = {}
    for col in deadvlive:
        d[col] = stats.mannwhitneyu(dead[col], live[col])[1]
    df_deadvlive = pd.DataFrame(d, index=[0])
    df_deadvlive.to_csv(os.path.join(TEST_ROOT, "deadvlive.csv"))

    return df_modelvlive, df_deadvlive


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
            if comparison == "bbvpy":
                d[col] = stats.mannwhitneyu(bb[col], py[col])[1]
            elif comparison == "bbvcv":
                d[col] = stats.mannwhitneyu(bb[col], cv[col])[1]
            elif comparison == "pyvcv":
                d[col] = stats.mannwhitneyu(py[col], cv[col])[1]
        df = pd.DataFrame(d, index=[0])
        df.to_csv(os.path.join(TEST_ROOT, f"{comparison}.csv"))
        all_dfs.append(df)

    return all_dfs[0], all_dfs[1], all_dfs[2]
