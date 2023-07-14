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
