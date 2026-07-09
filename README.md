# Ledger — Credit Scoring Prediction System

An end-to-end machine learning system that predicts an applicant's
creditworthiness (Creditworthy / Not Creditworthy) from financial and
behavioral data, served through a Flask web app with a live confidence
dial.

## Features

- Synthetic dataset generator modeled on the UCI/Statlog German Credit
  Dataset schema (age, income, employment, loan terms, credit history,
  savings, payment history, etc.) — 2,500+ rows with a genuine ~72/28
  class imbalance.
- Full EDA: shape/info, missing values, duplicates, statistical summary,
  distributions, correlation heatmap, class balance.
- Preprocessing: missing-value imputation, duplicate removal, one-hot +
  ordinal encoding, `StandardScaler` scaling.
- Feature engineering: debt-to-income ratio, repayment capacity, credit
  history score, savings score, income category, loan-to-income ratio,
  a composite high-risk flag.
- Class balancing with **SMOTE** (applied to the training split only).
- Four tuned models via `GridSearchCV` (5-fold, scored on ROC-AUC):
  Logistic Regression, Decision Tree, Random Forest, XGBoost. The best
  model is selected automatically and saved.
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix, classification report, model-comparison and ROC charts.
- Flask backend + a ledger/vault-themed frontend: applicant intake form
  on the left, a live confidence dial and approve/decline "stamp" on
  the right.

## Tech stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `imbalanced-learn` (SMOTE)
· `XGBoost` · `matplotlib` / `seaborn` · `Flask` · `joblib` · vanilla
HTML/CSS/JS frontend

## Project structure

```
Credit_Scoring_System/
├── app.py                     # Flask backend
├── train_model.py             # Full training pipeline
├── requirements.txt
├── README.md
├── dataset/
│   └── credit_data.csv
├── models/                    # created by train_model.py
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── feature_list.pkl
│   └── metrics.json
├── notebooks/
│   └── data_analysis.ipynb
├── static/
│   ├── style.css
│   └── images/                # EDA + evaluation charts, created by train_model.py
├── templates/
│   └── index.html
└── utils/
    ├── generate_dataset.py    # builds dataset/credit_data.csv
    └── preprocessing.py       # shared cleaning/feature-engineering/encoding
```

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Dataset

`dataset/credit_data.csv` is already included (2,500+ rows). To
regenerate it (or produce a different sample):

```bash
python utils/generate_dataset.py
```

## Training

```bash
python train_model.py
```

This will:
1. Load and analyze `dataset/credit_data.csv`, saving EDA charts to
   `static/images/`.
2. Clean, engineer features, encode, and scale the data.
3. Balance the training split with SMOTE.
4. Tune and evaluate Logistic Regression, Decision Tree, Random Forest,
   and XGBoost with `GridSearchCV`.
5. Pick the best model by test-set ROC-AUC and save it (plus the
   scaler, encoders, and feature list) into `models/`.
6. Save comparison/ROC/confusion-matrix/feature-importance charts to
   `static/images/`.

## Running the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Fill in the
applicant's details and click **Run assessment** to see the prediction,
confidence percentage, and risk tier.

## Notes

- `train_model.py` must be run at least once before `app.py`, since the
  app loads the artifacts from `models/`.
- The dataset is synthetic (see `utils/generate_dataset.py` for the
  generation logic) — it mirrors the *feature semantics and imbalance*
  of the real Statlog German Credit Dataset so the pipeline (SMOTE,
  encoding, tuning, evaluation) behaves realistically, without
  depending on external dataset downloads.
- To swap in the real Statlog German Credit dataset instead, replace
  `dataset/credit_data.csv` with a file using the same column names as
  those produced by `generate_dataset.py`, then re-run `train_model.py`.
