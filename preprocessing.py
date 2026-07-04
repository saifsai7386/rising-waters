"""
preprocessing.py — Data Pre-processing
----------------------------------------
- Handles missing values
- Detects & treats outliers (IQR capping)
- Encodes categorical values
- Splits dependent / independent variables
- Applies feature scaling
- Creates train / test datasets
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
FEATURE_COLS = MONTHS + ["ANNUAL"]  # SUBDIVISION/YEAR excluded from model features
TARGET_COL = "FLOODS"


def handle_missing_values(df):
    df = df.copy()
    for col in FEATURE_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def treat_outliers_iqr(df, cols):
    """Cap outliers to the IQR fences instead of dropping rows (keeps dataset size)."""
    df = df.copy()
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower, upper)
    return df


def encode_target(df):
    df = df.copy()
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])  # NO=0, YES=1 (alphabetical)
    return df, le


def load_and_clean(path):
    df = pd.read_csv(path)
    df = handle_missing_values(df)
    df = treat_outliers_iqr(df, FEATURE_COLS)
    df, label_encoder = encode_target(df)
    return df, label_encoder


def split_and_scale(df, test_size=0.2, random_state=42):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    import os
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rainfall_data.csv")
    df, le = load_and_clean(_path)
    print("Cleaned data shape:", df.shape)
    print("Missing values after cleaning:\n", df[FEATURE_COLS].isnull().sum().sum())
    print("Target classes:", dict(zip(le.classes_, le.transform(le.classes_))))

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
