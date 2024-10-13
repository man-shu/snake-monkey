# %%
# import libraries
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.plot_utils import (
    plot_cv_indices,
    chance_level,
    plot_confusion,
)
import seaborn as sns
import numpy as np

# %%
# read data
data_root = "data"
BB = pd.read_csv(os.path.join(data_root, "BB.csv"))
CV = pd.read_csv(os.path.join(data_root, "CV.csv"))
PY = pd.read_csv(os.path.join(data_root, "PY.csv"))

# concatenate data
df = pd.concat([BB, CV, PY], ignore_index=True)

# %%
# calculate feature
df["FR"] = df["FG"] + df["SS"]
# get classes and groups
df["snake model"] = np.select(
    [
        df["snake model"] == "BB",
        df["snake model"] == "CV",
        df["snake model"] == "PY",
    ],
    ["Bronze Back", "Pit Viper", "Python"],
    default=0,
)
classes = df["snake model"].to_numpy(dtype=object)
groups = df["subj"].to_numpy(dtype=object)
# drop unnecessary columns
df = df.drop(
    columns=["order", "sex", "VO", "TT", "FG", "SS", "snake model", "subj"]
)
# convert to numpy array
X = df.to_numpy()

# standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
# plotting params
sns.set(context="paper")
sns.set_style("white")

# dictionary to store results
results = {
    "accuracy": [],
    "auc": [],
    "predicted_labels": [],
    "expected_labels": [],
    "train_sets": [],
    "test_sets": [],
    # "weights": [],
}
# cross-validation
cv_splits = 100
cv = GroupShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
# cv = StratifiedShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
plot_cv_indices(cv.split(X, classes, groups), X, classes, groups, "plots")
# plot_cv_indices(cv.split(X, classes), X, classes, groups, "plots")
for train, test in cv.split(X, classes, groups):
    # for train, test in cv.split(X, classes):
    cf = SVC(kernel="rbf").fit(X[train], classes[train])
    predictions = cf.predict(X[test])
    accuracy = accuracy_score(classes[test], predictions)
    # auc = roc_auc_score(
    #     classes[test], cf.decision_function(X[test]), multi_class="ovo"
    # )
    results["accuracy"].append(accuracy)
    results["auc"].append(0)
    # store test labels and predictions
    results["predicted_labels"].append(predictions)
    results["expected_labels"].append(classes[test])
    results["train_sets"].append(train)
    results["test_sets"].append(test)
    # results["weights"].append(cf.coef_)

results_df = pd.DataFrame(results)
results_df = chance_level(results_df)
plot_confusion(results_df, "plots")

# %%
# import matplotlib.pyplot as plt

# sns.barplot(results_df["weights"])
# plt.show()
