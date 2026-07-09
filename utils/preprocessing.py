"""
preprocessing.py
-----------------
Shared preprocessing + feature-engineering logic used by BOTH
train_model.py (training time) and app.py (inference time), so the
exact same transformations are guaranteed to run in production as
were used during training.
"""

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["employment_status", "credit_history", "savings_account",
                     "housing", "purpose"]

NUMERIC_COLS = ["age", "income", "num_existing_loans", "loan_amount",
                 "loan_duration_months", "debt_ratio", "payment_history_score"]

# Ordinal mappings used for label-encoding columns that have a natural order
SAVINGS_ORDER = {"none": 0, "< $1000": 1, "$1000-$5000": 2,
                  "$5000-$10000": 3, "> $10000": 4}
CREDIT_HISTORY_ORDER = {"critical account": 0, "delay in past": 1, "no credits": 2,
                         "existing paid duly": 3, "all paid duly": 4}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicate rows."""
    df = df.drop_duplicates().reset_index(drop=True)

    # Numeric missing values -> median imputation
    for col in ["income", "payment_history_score"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical missing values -> mode imputation
    if "savings_account" in df.columns:
        df["savings_account"] = df["savings_account"].fillna(df["savings_account"].mode()[0])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived, risk-oriented features."""
    df = df.copy()

    # Debt-to-income ratio (annualized)
    df["debt_to_income_ratio"] = (df["loan_amount"] / df["loan_duration_months"]) / \
                                  (df["income"] / 12 + 1)

    # Loan repayment capacity: how much monthly income remains after the loan installment
    df["repayment_capacity"] = (df["income"] / 12) - (df["loan_amount"] / df["loan_duration_months"])

    # Credit history score (ordinal, higher = better)
    df["credit_history_score"] = df["credit_history"].map(CREDIT_HISTORY_ORDER)

    # Savings score (ordinal, higher = better cushion)
    df["savings_score"] = df["savings_account"].map(SAVINGS_ORDER)

    # Income category (bucketed)
    df["income_category"] = pd.cut(
        df["income"], bins=[0, 15000, 35000, 60000, np.inf],
        labels=["low", "lower-mid", "upper-mid", "high"]
    ).astype(str)

    # Loan-to-income ratio (overall exposure)
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["income"] + 1)

    # Composite risk flag: many existing loans + weak payment history
    df["high_risk_flag"] = ((df["num_existing_loans"] >= 3) &
                             (df["payment_history_score"] < 50)).astype(int)

    return df


def get_feature_columns():
    """Final feature list fed into the model (after encoding)."""
    return [
        "age", "income", "num_existing_loans", "loan_amount", "loan_duration_months",
        "debt_ratio", "payment_history_score", "debt_to_income_ratio",
        "repayment_capacity", "credit_history_score", "savings_score",
        "loan_to_income_ratio", "high_risk_flag",
        "employment_status", "housing", "purpose", "income_category",
    ]


def preprocess_pipeline(df: pd.DataFrame, fit_encoders=True, encoders=None):
    """
    Full preprocessing pipeline: clean -> engineer -> encode.
    Returns (X_dataframe, encoders_dict).
    `encoders` is a dict of {column: {category: code}} for one-hot columns
    built at train time and reused at inference time.
    """
    df = clean_data(df)
    df = engineer_features(df)

    onehot_cols = ["employment_status", "housing", "purpose", "income_category"]

    if fit_encoders:
        df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
        encoders = {col: sorted(df[col].unique().tolist()) for col in onehot_cols}
    else:
        # Ensure inference-time data has exactly the same one-hot columns as training
        df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
        for col, categories in encoders.items():
            for cat in categories:
                dummy_col = f"{col}_{cat}"
                if dummy_col not in df_encoded.columns:
                    df_encoded[dummy_col] = 0

    return df_encoded, encoders


def final_feature_list(encoders):
    """Builds the exact ordered list of model input columns given fitted encoders."""
    base = ["age", "income", "num_existing_loans", "loan_amount", "loan_duration_months",
            "debt_ratio", "payment_history_score", "debt_to_income_ratio",
            "repayment_capacity", "credit_history_score", "savings_score",
            "loan_to_income_ratio", "high_risk_flag"]
    onehot = []
    for col, categories in encoders.items():
        onehot += [f"{col}_{cat}" for cat in categories]
    return base + onehot
