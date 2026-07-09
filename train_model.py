
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from utils.preprocessing import preprocess_pipeline, final_feature_list, clean_data, engineer_features

sns.set_theme(style="darkgrid")
IMG_DIR = "static/images"


# ------------------------------------------------------------------ #
# 1. LOAD
# ------------------------------------------------------------------ #
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)
raw_df = pd.read_csv("dataset/credit_data.csv")
print(f"Shape: {raw_df.shape}")
print(raw_df.info())
print("\nMissing values:\n", raw_df.isnull().sum())
print(f"\nDuplicate rows: {raw_df.duplicated().sum()}")
print("\nStatistical summary:\n", raw_df.describe())


# ------------------------------------------------------------------ #
# 2. EDA VISUALIZATIONS
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# Class distribution
plt.figure(figsize=(6, 4.5))
counts = raw_df["creditworthy"].value_counts().sort_index()
plt.bar(["Not Creditworthy", "Creditworthy"], counts.values, color=["#E5484D", "#3FB68B"])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/class_distribution.png", dpi=120)
plt.close()

# Feature distributions
numeric_cols = ["age", "income", "loan_amount", "loan_duration_months",
                 "debt_ratio", "payment_history_score"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(raw_df[col].dropna(), kde=True, ax=ax, color="#3B82F6")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/feature_distributions.png", dpi=120)
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr = raw_df[numeric_cols + ["creditworthy"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/correlation_heatmap.png", dpi=120)
plt.close()

# Income vs credit risk
plt.figure(figsize=(7, 5))
sns.boxplot(data=raw_df, x="creditworthy", y="income", palette=["#E5484D", "#3FB68B"])
plt.xticks([0, 1], ["Not Creditworthy", "Creditworthy"])
plt.title("Income vs Credit Risk")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/income_vs_risk.png", dpi=120)
plt.close()

# Loan amount analysis
plt.figure(figsize=(7, 5))
sns.boxplot(data=raw_df, x="creditworthy", y="loan_amount", palette=["#E5484D", "#3FB68B"])
plt.xticks([0, 1], ["Not Creditworthy", "Creditworthy"])
plt.title("Loan Amount vs Credit Risk")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/loan_amount_analysis.png", dpi=120)
plt.close()

print(f"EDA charts saved to {IMG_DIR}/")


# ------------------------------------------------------------------ #
# 3-4. CLEAN, ENGINEER, ENCODE, SCALE
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("STEP 3-4: Cleaning, Feature Engineering, Encoding, Scaling")
print("=" * 60)

df = clean_data(raw_df)
df_encoded, encoders = preprocess_pipeline(df, fit_encoders=True)

feature_cols = final_feature_list(encoders)
X = df_encoded[feature_cols]
y = df_encoded["creditworthy"]

print(f"Final feature count: {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------------ #
# 5. SMOTE — balance the training set only
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("STEP 5: Balancing classes with SMOTE")
print("=" * 60)
print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("After SMOTE :", dict(pd.Series(y_train_res).value_counts()))


# ------------------------------------------------------------------ #
# 6. MODEL TRAINING + GRIDSEARCHCV
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("STEP 6: Training & tuning models")
print("=" * 60)

model_grids = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [4, 6, 8, 10], "min_samples_split": [2, 5, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [150, 250], "max_depth": [6, 10, None]},
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"n_estimators": [150, 250], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    ),
}

results = {}
fitted_models = {}

for name, (estimator, param_grid) in model_grids.items():
    print(f"\nTuning {name} ...")
    grid = GridSearchCV(estimator, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    best_model = grid.best_estimator_
    fitted_models[name] = best_model

    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        "best_params": grid.best_params_,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"  Best params: {grid.best_params_}")
    print(f"  Test ROC-AUC: {results[name]['roc_auc']}  |  F1: {results[name]['f1_score']}")


# ------------------------------------------------------------------ #
# 7. EVALUATION + COMPARISON CHARTS
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("STEP 7: Evaluation & model comparison")
print("=" * 60)

results_df = pd.DataFrame(results).T
print(results_df)

best_model_name = results_df["roc_auc"].astype(float).idxmax()
best_model = fitted_models[best_model_name]
print(f"\nBest model selected: {best_model_name}")

y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification report (best model):\n",
      classification_report(y_test, y_pred_best, target_names=["Not Creditworthy", "Creditworthy"]))

# Model comparison bar chart
plt.figure(figsize=(9, 5.5))
metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
results_df[metrics_to_plot].astype(float).plot(kind="bar", figsize=(9, 5.5), colormap="viridis")
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/model_comparison.png", dpi=120)
plt.close()

# Confusion matrix (best model)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Creditworthy", "Creditworthy"],
            yticklabels=["Not Creditworthy", "Creditworthy"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/confusion_matrix.png", dpi=120)
plt.close()

# ROC curve — all models
plt.figure(figsize=(7, 6))
for name, model in fitted_models.items():
    proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve — All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/roc_curve.png", dpi=120)
plt.close()

# Feature importance (tree-based only)
if best_model_name in ("Random Forest", "Decision Tree", "XGBoost"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", color="#C9A227")
    plt.title(f"Top Feature Importances — {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/feature_importance.png", dpi=120)
    plt.close()

print(f"Comparison charts saved to {IMG_DIR}/")


# ------------------------------------------------------------------ #
# 8. SAVE MODEL ARTIFACTS
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("STEP 8: Saving model artifacts")
print("=" * 60)

joblib.dump(best_model, "models/credit_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoder.pkl")
joblib.dump(feature_cols, "models/feature_list.pkl")

with open("models/metrics.json", "w") as f:
    json.dump({
        "best_model": best_model_name,
        "all_results": results,
    }, f, indent=2)

print("Saved: models/credit_model.pkl, scaler.pkl, encoder.pkl, feature_list.pkl, metrics.json")
print("\nTraining complete. Best model ->", best_model_name)
