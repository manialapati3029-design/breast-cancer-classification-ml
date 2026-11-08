"""
Train 5 classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluate them on a held-out test split, and save everything the app needs.

Run:  python train_models.py
Outputs:
    data/breast_cancer_full.csv   full dataset (for reference)
    test_data.csv                 held-out 20% test split (upload this to the app)
    model/*.joblib                one fitted Pipeline (scaler + model) per classifier
    model/metrics.csv             comparison table used in README
"""
import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)

RANDOM_STATE = 42
TARGET = "diagnosis"          # 1 = malignant, 0 = benign

# ---------- 1. Load data ----------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.rename(columns={"target": TARGET})
# sklearn encodes 0 = malignant, 1 = benign. Flip it so the positive class
# (1) is malignant, which is the class we care about detecting.
df[TARGET] = 1 - df[TARGET]

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
df.to_csv("data/breast_cancer_full.csv", index=False)

X = df.drop(columns=[TARGET])
y = df[TARGET]
print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print("Class balance:", y.value_counts().to_dict())

# ---------- 2. Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

test_df = X_test.copy()
test_df[TARGET] = y_test
test_df.to_csv("test_data.csv", index=False)
print(f"Train: {len(X_train)}  Test: {len(X_test)}  -> test_data.csv written")

# ---------- 3. Models (each wrapped with its scaler in one Pipeline) ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def evaluate(pipe, X, y):
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }, confusion_matrix(y, y_pred)


rows = []
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, os.path.join("model", FILE_NAMES[name]))
    metrics, cm = evaluate(pipe, X_test, y_test)
    rows.append({"ML Model Name": name, **metrics})
    print(f"\n{name}\n  CM [[TN FP][FN TP]] = {cm.tolist()}")

results = pd.DataFrame(rows).set_index("ML Model Name").round(4)
results.to_csv("model/metrics.csv")
print("\n", results.to_string())
