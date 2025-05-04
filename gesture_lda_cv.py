# %%
# import libraries
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from utils.plot_utils import (
    plot_cv_indices,
    chance_level,
    plot_confusion,
)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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
    default="0",
)
classes = df["snake model"].to_numpy(dtype=object)
groups = df["subj"].to_numpy(dtype=object)
# drop unnecessary columns
df = df.drop(
    columns=["order", "sex", "VO", "TT", "FG", "SS", "snake model", "subj"]
)
# convert to numpy array
X = df.to_numpy()

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
    "dummy_accuracy": [],
    "dummy_auc": [],
    "dummy_predicted_labels": [],
    # "weights": [],
}
# cross-validation
cv_splits = 100
cv = GroupShuffleSplit(n_splits=cv_splits, test_size=0.20, random_state=0)
# cv = StratifiedShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
plot_cv_indices(
    cv.split(X, classes, groups),
    X,
    classes,
    groups,
    "plots",
    filename="Gesture_LDA_cv_splits",
    figsize=(2, 14),
)  # plot_cv_indices(cv.split(X, classes), X, classes, groups, "plots")

# standardize data
scaler = StandardScaler()
cf = LinearDiscriminantAnalysis()
dummy = DummyClassifier()
scale_cf = make_pipeline(scaler, cf)
scale_dummy = make_pipeline(scaler, dummy)

for train, test in cv.split(X, classes, groups):
    # for train, test in cv.split(X, classes):

    # fit model
    scale_cf.fit(X[train], classes[train])
    scale_dummy.fit(X[train], classes[train])
    # make predictions
    predictions = scale_cf.predict(X[test])
    dummy_predictions = scale_dummy.predict(X[test])
    # calculate accuracy and auc
    accuracy = accuracy_score(classes[test], predictions)
    auc = roc_auc_score(
        classes[test], scale_cf.predict_proba(X[test]), multi_class="ovr"
    )
    dummy_accuracy = accuracy_score(classes[test], dummy_predictions)
    dummy_auc = roc_auc_score(
        classes[test], scale_dummy.predict_proba(X[test]), multi_class="ovr"
    )
    # store results
    results["accuracy"].append(accuracy)
    results["auc"].append(auc)
    results["dummy_accuracy"].append(dummy_accuracy)
    results["dummy_auc"].append(dummy_auc)
    # store test labels and predictions
    results["predicted_labels"].append(predictions)
    results["expected_labels"].append(classes[test])
    results["train_sets"].append(train)
    results["test_sets"].append(test)
    # store dummy predictions
    results["dummy_predicted_labels"].append(dummy_predictions)

results_df = pd.DataFrame(results)
results_df = chance_level(results_df)
# multiply all floats by 100
for column in results_df.columns:
    if results_df[column].dtype == float:
        results_df[column] *= 100
results_df.to_csv("results/Gesture_LDA_cv_results_full.csv", index=False)
av_results_df = results_df.drop(
    columns=[
        "train_sets",
        "test_sets",
        "predicted_labels",
        "expected_labels",
        "labels",
        "dummy_predicted_labels",
    ]
)
std = av_results_df.std()
av_results_df = av_results_df.mean()
av_results_df = pd.concat([av_results_df, std], axis=1)
av_results_df.rename(columns={0: "mean", 1: "std"}, inplace=True)
av_results_df.to_csv("results/Gesture_LDA_cv_results_average.csv")

# save classification report
report = classification_report(
    np.concatenate(results_df["expected_labels"].to_numpy()),
    np.concatenate(results_df["predicted_labels"].to_numpy()),
    digits=4,
)
# write to txt file
with open("results/Gesture_LDA_cv_report.txt", "w") as f:
    f.write(report)

dummy_report = classification_report(
    np.concatenate(results_df["expected_labels"].to_numpy()),
    np.concatenate(results_df["dummy_predicted_labels"].to_numpy()),
    digits=4,
)
# write to txt file
with open("results/Gesture_LDA_cv_dummy_report.txt", "w") as f:
    f.write(dummy_report)

plot_confusion(
    results_df,
    "plots",
    "Gesture_LDA_cv_confusion",
    chance_level=results_df["chance"].mean(),
)

# %%
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
plt.savefig("plots/Gesture_LDA_cv_accuracy.png", bbox_inches="tight", dpi=600)
plt.savefig("plots/Gesture_LDA_cv_accuracy.tiff", bbox_inches="tight", dpi=600)
plt.close()
