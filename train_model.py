"""
train_model.py — Model Building, Evaluation & Best Model Selection
---------------------------------------------------------------------
Trains Decision Tree, Random Forest, KNN, and XGBoost classifiers,
evaluates each with confusion matrix / classification report / accuracy,
picks the best performer, and saves the (model, scaler, feature_cols)
bundle to models/floods.save for the Flask app.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

from preprocessing import load_and_clean, split_and_scale, FEATURE_COLS

import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "data", "rainfall_data.csv")
MODEL_SAVE_PATH = os.path.join(_BASE_DIR, "models", "floods.save")
PLOT_DIR = os.path.join(_BASE_DIR, "notebooks", "plots")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["NO", "YES"])

    print(f"\n{'='*55}\nModel: {name}\n{'='*55}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NO", "YES"], yticklabels=["NO", "YES"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(f"{PLOT_DIR}/confusion_matrix_{safe_name}.png", dpi=120)
    plt.close()

    return acc


def main():
    df, label_encoder = load_and_clean(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    models = get_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = evaluate_model(name, model, X_test, y_test)
        results[name] = (acc, model)

    print(f"\n{'='*55}\nModel Comparison Summary\n{'='*55}")
    summary_df = pd.DataFrame(
        {name: [acc] for name, (acc, _) in results.items()}, index=["Accuracy"]
    ).T.sort_values("Accuracy", ascending=False)
    print(summary_df)

    plt.figure(figsize=(7, 4))
    sns.barplot(x=summary_df.index, y=summary_df["Accuracy"], palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/model_accuracy_comparison.png", dpi=120)
    plt.close()

    # If XGBoost is within 1% accuracy of the top score, prefer it — it's
    # generally the more robust choice on unseen tabular data, which is
    # why it's the model this project deploys.
    top_acc = summary_df["Accuracy"].iloc[0]
    if results["XGBoost"][0] >= top_acc - 0.01:
        best_name = "XGBoost"
    else:
        best_name = summary_df.index[0]
    best_acc, best_model = results[best_name]
    print(f"\nBest model: {best_name} (Accuracy: {best_acc:.4f})")

    bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "label_encoder": label_encoder,
        "model_name": best_name,
        "accuracy": best_acc,
    }
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved best model bundle to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
