"""
app.py — Flask Web Application
--------------------------------
Pages:
  /              Home page
  /predict       Prediction input form (GET) and handles submission (POST)
  /result        Flood chance / No flood chance result page (rendered from /predict POST)
"""

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

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


@app.route("/")
def home():
    return render_template("home.html", model_name=model_name,
                            accuracy=round(model_accuracy * 100, 2))


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

        annual = sum(monthly_values)
        features = np.array(monthly_values + [annual]).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = round(max(proba) * 100, 2)

        input_summary = dict(zip(MONTHS, monthly_values))
        input_summary["ANNUAL"] = round(annual, 2)

        if prediction == 1:
            return render_template(
                "flood_result.html",
                confidence=confidence,
                inputs=input_summary,
                model_name=model_name,
            )
        else:
            return render_template(
                "no_flood_result.html",
                confidence=confidence,
                inputs=input_summary,
                model_name=model_name,
            )
    except Exception as e:
        return render_template("predict.html", months=MONTHS,
                                error=f"Please check your input values. ({e})")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
