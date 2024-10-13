import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import (
    cross_validate,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.plot_utils import plot_cv_indices, chance_level, plot_confusion
import matplotlib.pyplot as plt
import seaborn as sns

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

# groups
groups = snakeCalls["subj"]
n_groups = len(np.unique(groups))

results = {
    "accuracy": [],
    "auc": [],
    "predicted_labels": [],
    "expected_labels": [],
    "train_sets": [],
    "test_sets": [],
    "dummy_accuracy": [],
    "dummy_auc": [],
}
# cross-validation
cv_splits = 100
cv = StratifiedShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=42)
plot_cv_indices(
    cv.split(X, y, groups),
    X,
    y,
    groups,
    "plots",
    filename="Vocal_LDA_cv_splits",
    figsize=(2, 14),
)
for train, test in cv.split(X, y, groups):
    cf = LinearDiscriminantAnalysis().fit(X[train], y[train])
    dummy = DummyClassifier().fit(X[train], y[train])
    predictions = cf.predict(X[test])
    accuracy = accuracy_score(y[test], predictions)
    auc = roc_auc_score(y[test], cf.predict_proba(X[test]), multi_class="ovr")
    dummy_predictions = dummy.predict(X[test])
    dummy_accuracy = accuracy_score(y[test], dummy_predictions)
    dummy_auc = roc_auc_score(
        y[test], dummy.predict_proba(X[test]), multi_class="ovr"
    )
    results["accuracy"].append(accuracy)
    results["auc"].append(auc)
    results["dummy_accuracy"].append(dummy_accuracy)
    results["dummy_auc"].append(dummy_auc)
    # store test labels and predictions
    results["predicted_labels"].append(predictions)
    results["expected_labels"].append(y[test])
    results["train_sets"].append(train)
    results["test_sets"].append(test)

results_df = pd.DataFrame(results)
results_df = chance_level(results_df)
plot_confusion(
    results_df,
    "plots",
    "Vocal_LDA_cv_confusion",
    chance_level=results_df["chance"].mean(),
)

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(
    ["LDA", "Dummy", "Chance"],
    [
        np.mean(results_df["accuracy"]),
        np.mean(results_df["dummy_accuracy"]),
        np.mean(results_df["chance"]),
    ],
    yerr=[
        np.std(results_df["accuracy"]) / np.sqrt(cv_splits),
        np.std(results_df["dummy_accuracy"]) / np.sqrt(cv_splits),
        np.std(results_df["chance"]) / np.sqrt(cv_splits),
    ],
)
plt.ylabel("Accuracy")
sns.despine(top=True, right=True, ax=ax)
plt.savefig("plots/Vocal_LDA_cv_accuracy.png", bbox_inches="tight", dpi=600)
plt.savefig("plots/Vocal_LDA_cv_accuracy.tiff", bbox_inches="tight", dpi=600)
plt.close()
