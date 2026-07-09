

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from utils.preprocessing import preprocess_pipeline, final_feature_list

app = Flask(__name__)


try:
    model = joblib.load("models/credit_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoder.pkl")
    feature_cols = joblib.load("models/feature_list.pkl")
    ARTIFACTS_LOADED = True
except FileNotFoundError:
    # Lets the app boot even before training has been run, with a clear error surfaced to the UI
    model = scaler = encoders = feature_cols = None
    ARTIFACTS_LOADED = False


FORM_FIELDS = {
    "age": int,
    "income": float,
    "employment_status": str,
    "loan_amount": float,
    "loan_duration_months": int,
    "num_existing_loans": int,
    "debt_ratio": float,
    "credit_history": str,
    "savings_account": str,
    "payment_history_score": float,
    "housing": str,
    "purpose": str,
}


@app.route("/")
def index():
    return render_template("index.html", artifacts_loaded=ARTIFACTS_LOADED)


@app.route("/predict", methods=["POST"])
def predict():
    if not ARTIFACTS_LOADED:
        return jsonify({
            "error": "Model artifacts not found. Please run 'python train_model.py' first."
        }), 500

    try:
        data = request.get_json() if request.is_json else request.form

        row = {}
        for field, caster in FORM_FIELDS.items():
            value = data.get(field)
            if value is None or value == "":
                return jsonify({"error": f"Missing field: {field}"}), 400
            row[field] = caster(value)

        input_df = pd.DataFrame([row])

        # Run through the exact same pipeline used at training time
        encoded_df, _ = preprocess_pipeline(input_df, fit_encoders=False, encoders=encoders)

        # Align columns exactly to the training-time feature list
        for col in feature_cols:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
        encoded_df = encoded_df[feature_cols]

        scaled = scaler.transform(encoded_df)

        prediction = int(model.predict(scaled)[0])
        probability = float(model.predict_proba(scaled)[0][1])  # P(Creditworthy)

        result = {
            "prediction": prediction,
            "label": "Creditworthy" if prediction == 1 else "Not Creditworthy",
            "probability": round(probability * 100, 1),
            "risk_level": (
                "Low Risk" if probability >= 0.75 else
                "Moderate Risk" if probability >= 0.5 else
                "High Risk"
            ),
        }
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
