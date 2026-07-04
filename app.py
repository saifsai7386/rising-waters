"""
app.py — Flask Web Application
--------------------------------
Pages:
  /              Home page
  /predict       Prediction input form (GET) and handles submission (POST)
  /result        Flood chance / No flood chance result page (rendered from /predict POST)
"""

import csv
import io
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, session, Response, url_for, redirect

app = Flask(__name__)
app.secret_key = "replace_this_with_a_secure_key"

MODEL_PATH = "models/floods.save"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]
model_name = bundle["model_name"]
model_accuracy = bundle["accuracy"]

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

MONTH_THRESHOLDS = {
    "JAN": 13.8,
    "FEB": 18.6,
    "MAR": 26.6,
    "APR": 63.9,
    "MAY": 182.7,
    "JUN": 765.6,
    "JUL": 1095.2,
    "AUG": 583.3,
    "SEP": 380.7,
    "OCT": 248.8,
    "NOV": 93.7,
    "DEC": 20.7,
}


def get_likely_flood_months(monthly_values):
    months = []
    for idx, value in enumerate(monthly_values):
        month = MONTHS[idx]
        threshold = MONTH_THRESHOLDS.get(month, 0)
        if value >= threshold:
            months.append(month)
    return months


def build_risk_band(prediction, confidence, annual):
    if prediction == 1:
        if confidence >= 85 or annual >= 3300:
            return "High"
        if confidence >= 70 or annual >= 3000:
            return "Moderate"
        return "Elevated"
    if annual >= 3000 or confidence >= 65:
        return "Moderate"
    return "Low"


def build_rainfall_summary(monthly_values):
    peak_idx = int(np.argmax(monthly_values))
    low_idx = int(np.argmin(monthly_values))
    return {
        "peak_month": MONTHS[peak_idx],
        "peak_value": round(monthly_values[peak_idx], 1),
        "lowest_month": MONTHS[low_idx],
        "lowest_value": round(monthly_values[low_idx], 1),
    }


def build_recommendation(prediction, risk_band, confidence):
    if prediction == 1:
        if risk_band == "High":
            return "Urgent flood preparedness is recommended — monitor local alerts and move valuables to higher ground."
        if risk_band == "Moderate":
            return "Stay alert and prepare emergency supplies. Review evacuation routes in case conditions worsen."
        return "Watch weather updates closely and avoid low-lying areas until the storm passes."

    if confidence >= 60 or risk_band == "Moderate":
        return "Current readings are safe, but keep monitoring forecasts through the season."
    return "No immediate flood concern. Continue routine observation and check again if rainfall increases."


def save_prediction_history(entry):
    history = session.get("history", [])
    history.insert(0, entry)
    session["history"] = history[:5]


@app.route("/")
def home():
    history = session.get("history", [])
    return render_template(
        "home.html",
        model_name=model_name,
        accuracy=round(model_accuracy * 100, 2),
        history=history,
        recent_count=len(history),
    )


@app.route("/download-history")
def download_history():
    history = session.get("history", [])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Prediction", "Confidence", "Risk Band", "Annual"])
    for item in history:
        writer.writerow([
            item["timestamp"],
            item["prediction"],
            item["confidence"],
            item["risk_band"],
            item["annual"],
        ])
    resp = Response(output.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = "attachment; filename=recent_flood_checks.csv"
    return resp


@app.route("/clear-history")
def clear_history():
    session.pop("history", None)
    return redirect(url_for("home"))


@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html", months=MONTHS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        monthly_values = []
        for month in MONTHS:
            val = float(request.form.get(month, 0))
            monthly_values.append(val)

        cloud_visibility = float(request.form.get("CLOUD_VISIBILITY", 0))
        annual = sum(monthly_values)
        features = pd.DataFrame(
            [monthly_values + [annual, cloud_visibility]],
            columns=feature_cols,
        )
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(round(max(proba) * 100, 2))

        input_summary = dict(zip(MONTHS, monthly_values))
        input_summary["ANNUAL"] = round(annual, 2)
        input_summary["CLOUD_VISIBILITY"] = round(cloud_visibility, 1)
        month_risks = get_likely_flood_months(monthly_values)
        risk_band = build_risk_band(prediction, confidence, annual)
        rainfall_summary = build_rainfall_summary(monthly_values)
        recommendation = build_recommendation(prediction, risk_band, confidence)
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "prediction": "Flood" if prediction == 1 else "No Flood",
            "confidence": confidence,
            "risk_band": risk_band,
            "annual": float(round(annual, 2)),
        }
        save_prediction_history(history_entry)

        template_data = {
            "confidence": confidence,
            "inputs": input_summary,
            "model_name": model_name,
            "month_risks": month_risks,
            "risk_band": risk_band,
            "rainfall_summary": rainfall_summary,
            "recommendation": recommendation,
        }

        if prediction == 1:
            return render_template("flood_result.html", **template_data)
        else:
            return render_template("no_flood_result.html", **template_data)
    except Exception as e:
        return render_template("predict.html", months=MONTHS,
                                error=f"Please check your input values. ({e})")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
