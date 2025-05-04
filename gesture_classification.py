# %%
# import libraries
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.svm import SVC
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
from itertools import product

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

models = ["SVC", "LDA"]
cv_n_splits = [10, 50, 100, 200, 500]
test_sizes = [0.4, 0.3, 0.2]
model_to_obj = {"SVC": SVC(), "LDA": LinearDiscriminantAnalysis()}
overall_results = {"SVC": [], "LDA": [], "cv": [], "test": []}


for model, cv_n_split, test_size in product(models, cv_n_splits, test_sizes):
    name_str = f"model-{model}_cv-{cv_n_split}_test-{test_size}"
    print(name_str)
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
    cv = GroupShuffleSplit(
        n_splits=cv_n_split, test_size=test_size, random_state=0
    )
    plot_cv_indices(
        cv.split(X, classes, groups),
        X,
        classes,
        groups,
        "plots",
        filename=f"Gesture_{name_str}_crosvalsplits",
        figsize=(2, 14),
    )
    # standardize data
    scaler = StandardScaler()
    cf = model_to_obj[model]
    dummy = DummyClassifier()
    scale_cf = make_pipeline(scaler, cf)
    scale_dummy = make_pipeline(scaler, dummy)

    for train, test in cv.split(X, classes, groups):
        # fit model
        scale_cf.fit(X[train], classes[train])
        scale_dummy.fit(X[train], classes[train])
        # make predictions
        predictions = scale_cf.predict(X[test])
        dummy_predictions = scale_dummy.predict(X[test])
        # calculate accuracy and auc
        accuracy = accuracy_score(classes[test], predictions)
        if model == "SVC":
            auc = 0
        else:
            # for LDA, use the decision function instead of predict_proba
            auc = roc_auc_score(
                classes[test],
                scale_cf.predict_proba(X[test]),
                multi_class="ovr",
            )
        dummy_accuracy = accuracy_score(classes[test], dummy_predictions)
        dummy_auc = roc_auc_score(
            classes[test],
            scale_dummy.predict_proba(X[test]),
            multi_class="ovr",
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

    results_df.to_csv(
        f"results/Gesture_{name_str}_results_full.csv",
        index=False,
    )
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
    av_results_df.to_csv(f"results/Gesture_{name_str}_results_average.csv")

    # save classification report
    report = classification_report(
        np.concatenate(results_df["expected_labels"].to_numpy()),
        np.concatenate(results_df["predicted_labels"].to_numpy()),
        digits=4,
        zero_division=0,
    )
    # write to txt file
    with open(f"results/Gesture_{name_str}_report.txt", "w") as f:
        f.write(report)

    dummy_report = classification_report(
        np.concatenate(results_df["expected_labels"].to_numpy()),
        np.concatenate(results_df["dummy_predicted_labels"].to_numpy()),
        digits=4,
        zero_division=0,
    )
    # write to txt file
    with open(f"results/Gesture_{name_str}_dummyreport.txt", "w") as f:
        f.write(dummy_report)

    plot_confusion(
        results_df,
        "plots",
        f"Gesture_{name_str}_confusion",
        chance_level=results_df["chance"].mean(),
    )

    # %%
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(
        [model, "Dummy", "Chance"],
        [
            av_results_df.loc["accuracy", "mean"],
            av_results_df.loc["dummy_accuracy", "mean"],
            av_results_df.loc["chance", "mean"],
        ],
        yerr=[
            av_results_df.loc["accuracy", "std"],
            av_results_df.loc["dummy_accuracy", "std"],
            av_results_df.loc["chance", "std"],
        ],
    )
    plt.ylabel("Accuracy (%)")
    sns.despine(top=True, right=True, ax=ax)
    plt.savefig(
        f"plots/Gesture{name_str}_accuracy.png", bbox_inches="tight", dpi=600
    )
    plt.savefig(
        f"plots/Gesture_{name_str}_accuracy.tiff", bbox_inches="tight", dpi=600
    )
    plt.close()

    # store results
    overall_results[model].append(av_results_df.loc["accuracy", "mean"])
    overall_results["cv"].append(cv_n_split)
    overall_results["test"].append(test_size)

overall_results["cv"] = overall_results["cv"][
    0 : int(len(overall_results["cv"]) / 2)
]
overall_results["test"] = overall_results["test"][
    0 : int(len(overall_results["test"]) / 2)
]
overall_results["test"] = [x * 100 for x in overall_results["test"]]
# save overall results
overall_results_df = pd.DataFrame(overall_results)
overall_results_df.to_csv("results/Gesture_overall.csv", index=False)
