# Breast Cancer Diagnosis – Comparison of Five Classification Models

**Live app:** https://breast-cancer-classification-ml-manialapati3029.streamlit.app/

## a. Problem statement

Given 30 numeric measurements of cell nuclei taken from a digitised fine-needle aspirate (FNA) image of a breast mass, predict whether the mass is **malignant (1)** or **benign (0)**. This is a binary classification problem. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data, evaluated on the same held-out test set with six metrics, and served through an interactive Streamlit app that loads the saved models and evaluates them on uploaded test data without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (loaded via `sklearn.datasets.load_breast_cancer`, which ships the same UCI data) |
| Instances | 569 |
| Features | 30 (all continuous, no missing values) |
| Target | `diagnosis`: 1 = malignant (212), 0 = benign (357) |
| Class balance | 37.3 % malignant / 62.7 % benign (mildly imbalanced) |

The 30 features are 10 base measurements of each cell nucleus (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each reported three ways: mean, standard error, and "worst" (mean of the three largest values).

Note: scikit-learn encodes benign as 1. The training script flips this so that malignant is the positive class, because precision, recall and F1 should describe how well the model detects cancer.

**Split:** stratified 80 / 20 train–test split, `random_state=42` → 455 training rows, 114 test rows. The 114 test rows are saved as `test_data.csv`.

**Preprocessing:** `StandardScaler`, fitted on training data only and saved inside each model's `Pipeline`, so the app applies exactly the same scaling.

## c. GitHub repository link

https://github.com/manialapati3029-design/breast-cancer-classification-ml/tree/develop

Repository structure:

```
project-folder/
│-- app.py                     Streamlit app (loads saved models, no retraining)
│-- train_models.py            Trains all 5 models, saves them and test_data.csv
│-- requirements.txt
│-- README.md
│-- test_data.csv              Held-out test split (114 rows, 30 features + target)
│-- data/breast_cancer_full.csv
│-- .streamlit/config.toml     App theme
│-- model/
│     logistic_regression.joblib
│     decision_tree.joblib
│     knn.joblib
│     naive_bayes.joblib
│     svm.joblib
│     metrics.csv
```

## d. Models used

All five models are scikit-learn `Pipeline([StandardScaler, model])` objects:

- Logistic Regression – `LogisticRegression(max_iter=1000)`
- Decision Tree – `DecisionTreeClassifier()` (no depth limit)
- KNN – `KNeighborsClassifier(n_neighbors=5)`
- Naive Bayes – `GaussianNB()` (Gaussian chosen because features are continuous and become negative after scaling, which Multinomial NB cannot accept)
- Support Vector Machine – `SVC(kernel="rbf", probability=True)`

### Comparison table (test set, n = 114)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

Confusion matrices on the test set (TN, FP, FN, TP): LR (71, 1, 3, 39) · DT (68, 4, 4, 38) · KNN (71, 1, 4, 38) · NB (69, 3, 6, 36) · SVM (72, 0, 3, 39).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best overall (accuracy 0.9649, MCC 0.9245) and the **highest AUC (0.9960)**, so it ranks malignant above benign cases almost perfectly. A linear boundary is enough because, after scaling, the size- and shape-related features separate the two classes almost linearly. It made 1 false positive and missed 3 malignant cases. It is also the most interpretable model here. |
| Decision Tree | Weakest ranking ability (**lowest AUC, 0.9246**) and second-lowest MCC (0.8492). The unpruned tree grows until its leaves are pure, so it overfits the 455 training rows and outputs almost only 0/1 probabilities, which is why its AUC is barely above its accuracy. Its errors are split evenly (4 FP, 4 FN). Limiting `max_depth` or `min_samples_leaf` would likely help. |
| KNN | Solid (accuracy 0.9561, MCC 0.9058) with high precision (0.9744) but lower recall (0.9048): 4 malignant cases were missed. KNN depends entirely on distances, so scaling is essential here; without it, the large-valued `area` features would dominate. |
| Naive Bayes | **Lowest accuracy (0.9211), recall (0.8571) and MCC (0.8292)**: it missed 6 malignant tumours, the most of any model. Its AUC is still high (0.9891), so it ranks cases well but its 0.5 cut-off is poorly placed. The cause is its independence assumption: radius, perimeter and area are almost perfectly correlated, so the model counts the same evidence several times and becomes overconfident. |
| Support Vector Machine | **Best on accuracy (0.9737), F1 (0.9630) and MCC (0.9442)**, with **perfect precision (1.0000)**: no benign case was flagged as malignant. It missed 3 malignant cases, the same as Logistic Regression. The RBF kernel with scaled features gives a smooth, well-regularised boundary on this small, clean dataset. |
| **Overall winner for this dataset** | **Support Vector Machine**, by MCC, accuracy and F1. Logistic Regression is a very close second and has the best AUC. With only 114 test rows, the gap between them is one patient, so they should be treated as roughly tied. For a cancer-screening use case, recall (missed cancers) matters more than precision; both models miss 3 of 42 malignant cases, and lowering their decision threshold would trade a few false alarms for fewer missed cancers. |

## Streamlit app features

1. CSV upload of test data (plus a download button and a "use bundled test_data.csv" option)
2. Model selection dropdown (5 models)
3. Display of Accuracy, AUC, Precision, Recall, F1 and MCC
4. Confusion matrix and classification report
5. Side-by-side comparison of all five models on the uploaded data

The app only loads the saved `.joblib` pipelines and calls `predict` / `predict_proba`. It never retrains.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates models, test_data.csv, metrics
streamlit run app.py
```
