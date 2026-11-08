"""
Streamlit app: evaluate 5 pre-trained classifiers on uploaded test data.
Models are loaded from model/*.joblib. Nothing is retrained here.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

TARGET = "diagnosis"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Breast cancer classifier comparison", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(os.path.join(MODEL_DIR, f)) for name, f in MODEL_FILES.items()}


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


models = load_models()
expected_features = list(next(iter(models.values())).feature_names_in_)

# ---------- Header ----------
st.title("Breast cancer diagnosis: five classifiers, one test set")
st.write(
    "Upload the held-out test data and see how each pre-trained model performs. "
    "Dataset: Breast Cancer Wisconsin (Diagnostic), UCI. "
    "Target column `diagnosis`: 1 = malignant, 0 = benign."
)

# ---------- Sidebar: upload + model choice ----------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f:
            st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=False)

    st.header("2. Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

# ---------- Load data ----------
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample and os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
else:
    st.info("Upload a test CSV in the sidebar, or tick “Use the bundled test_data.csv”.")
    st.stop()

missing = [c for c in expected_features + [TARGET] if c not in df.columns]
if missing:
    st.error(
        f"The CSV is missing {len(missing)} required column(s): {', '.join(missing[:8])}"
        + (" …" if len(missing) > 8 else "")
        + ". Use test_data.csv from the repository as the template."
    )
    st.stop()

X = df[expected_features]
y = df[TARGET].astype(int)
st.caption(f"Loaded {len(df)} rows · {len(expected_features)} features · "
           f"{int(y.sum())} malignant, {int((y == 0).sum())} benign")

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), width="stretch")

# ---------- Selected model ----------
st.subheader(model_name)
metrics, y_pred = compute_metrics(models[model_name], X, y)

cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---------- All models side by side ----------
st.subheader("All models on this test data")
rows = []
for name, m in models.items():
    met, _ = compute_metrics(m, X, y)
    rows.append({"ML Model Name": name, **met})
table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(table.style.highlight_max(axis=0, color="#D6E2F3"), width="stretch")
best = table["MCC"].idxmax()
st.write(f"Highest MCC on this data: **{best}** ({table.loc[best, 'MCC']:.4f}).")
