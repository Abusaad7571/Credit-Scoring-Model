"""
generate_dataset.py
--------------------
Generates a realistic, synthetic credit-scoring dataset modeled on the
structure and feature semantics of the UCI/Statlog German Credit Dataset.

Why synthetic? The original Statlog file isn't reachable from this
environment, so we recreate its feature semantics (age, employment,
loan amount/duration, savings, credit history, etc.) with realistic
statistical relationships and inject a genuine class imbalance
(~72% Creditworthy / 28% Not Creditworthy) so SMOTE has real work to do.

Run:
    python utils/generate_dataset.py
Produces:
    dataset/credit_data.csv   (2500 rows)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 2500


def generate():
    age = RNG.integers(19, 70, N)

    employment_status = RNG.choice(
        ["employed", "self-employed", "unemployed", "student", "retired"],
        size=N, p=[0.55, 0.15, 0.10, 0.08, 0.12]
    )

    base_income = {
        "employed": 45000, "self-employed": 52000, "unemployed": 9000,
        "student": 8000, "retired": 24000,
    }
    income = np.array([
        max(4000, RNG.normal(base_income[e], base_income[e] * 0.35))
        for e in employment_status
    ]).round(2)

    credit_history = RNG.choice(
        ["no credits", "all paid duly", "existing paid duly",
         "delay in past", "critical account"],
        size=N, p=[0.10, 0.25, 0.35, 0.18, 0.12]
    )

    savings_account = RNG.choice(
        ["none", "< $1000", "$1000-$5000", "$5000-$10000", "> $10000"],
        size=N, p=[0.22, 0.33, 0.25, 0.13, 0.07]
    )

    num_existing_loans = RNG.poisson(1.1, N).clip(0, 6)

    loan_amount = np.round(RNG.uniform(500, 20000, N) * (1 + num_existing_loans * 0.05), 2)
    loan_duration = RNG.choice([6, 12, 18, 24, 36, 48, 60], size=N,
                               p=[0.10, 0.20, 0.15, 0.20, 0.15, 0.12, 0.08])

    # Debt ratio: monthly-debt-ish proxy relative to income, shaped by loan size/duration
    debt_ratio = np.clip(
        (loan_amount / (loan_duration * (income / 12 + 1))) * RNG.uniform(0.5, 1.3, N),
        0.01, 1.5
    ).round(3)

    payment_history_score = np.clip(
        RNG.normal(70, 18, N)
        - (credit_history == "delay in past") * 20
        - (credit_history == "critical account") * 35
        + (credit_history == "all paid duly") * 12,
        0, 100
    ).round(1)

    housing = RNG.choice(["own", "rent", "free"], size=N, p=[0.55, 0.35, 0.10])
    purpose = RNG.choice(
        ["car", "furniture", "education", "business", "appliances", "other"],
        size=N, p=[0.28, 0.18, 0.15, 0.14, 0.13, 0.12]
    )

    # ---- Latent creditworthiness score (drives the label, not observed directly) ----
    savings_map = {"none": 0, "< $1000": 1, "$1000-$5000": 2, "$5000-$10000": 3, "> $10000": 4}
    credit_hist_map = {"no credits": 1, "all paid duly": 4, "existing paid duly": 3,
                        "delay in past": -2, "critical account": -4}
    employment_map = {"employed": 3, "self-employed": 2, "unemployed": -3,
                       "student": 0, "retired": 1}

    latent = (
        0.00006 * income
        - 1.8 * debt_ratio
        + 0.04 * payment_history_score
        + np.array([savings_map[s] for s in savings_account]) * 1.3
        + np.array([credit_hist_map[c] for c in credit_history]) * 1.5
        + np.array([employment_map[e] for e in employment_status]) * 1.2
        - num_existing_loans * 1.1
        - loan_amount / 8000
        + RNG.normal(0, 3.2, N)              # noise
    )

    threshold = np.percentile(latent, 28)     # ~28% negative class -> real imbalance
    target = (latent > threshold).astype(int)  # 1 = Creditworthy, 0 = Not Creditworthy

    df = pd.DataFrame({
        "age": age,
        "employment_status": employment_status,
        "income": income,
        "credit_history": credit_history,
        "savings_account": savings_account,
        "num_existing_loans": num_existing_loans,
        "loan_amount": loan_amount,
        "loan_duration_months": loan_duration,
        "debt_ratio": debt_ratio,
        "payment_history_score": payment_history_score,
        "housing": housing,
        "purpose": purpose,
        "creditworthy": target,
    })

    # Sprinkle a small, realistic amount of missingness + a few duplicate rows
    for col in ["income", "payment_history_score", "savings_account"]:
        mask = RNG.random(N) < 0.02
        df.loc[mask, col] = np.nan

    dupes = df.sample(15, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv("dataset/credit_data.csv", index=False)
    print(f"Saved dataset/credit_data.csv -> {data.shape[0]} rows, {data.shape[1]} columns")
    print(data["creditworthy"].value_counts(normalize=True).round(3))
