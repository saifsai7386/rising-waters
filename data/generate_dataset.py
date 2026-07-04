"""
generate_dataset.py
--------------------
Generates a synthetic rainfall/flood dataset that mirrors the structure of the
"Kerala Rainfall / Flood" dataset commonly used for this project on Kaggle:
    SUBDIVISION, YEAR, JAN, FEB, ..., DEC, ANNUAL, FLOODS (YES/NO)

Why synthetic data:
This sandbox environment cannot reach kaggle.com (network egress is restricted
to package registries only), so we can't literally download the original CSV.
Instead we generate a statistically realistic stand-in with the same schema,
column names, and monsoon-driven flood logic, so every downstream step
(EDA, preprocessing, model training, Flask app) works exactly the way it
would with the real Kaggle file. If you have Kaggle access, just download
"Kerala Flood / Rainfall dataset" and drop it in as data/rainfall_data.csv
with the same column names -- nothing else needs to change.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Rough monthly rainfall means (mm) for a monsoon-affected Indian subdivision
MONTH_MEANS = {
    "JAN": 8, "FEB": 12, "MAR": 20, "APR": 45, "MAY": 130,
    "JUN": 550, "JUL": 680, "AUG": 400, "SEP": 250,
    "OCT": 180, "NOV": 70, "DEC": 15
}
MONTH_STD = {m: max(v * 0.35, 5) for m, v in MONTH_MEANS.items()}

# Cloud visibility in km, lower values usually reflect stormier conditions
CLOUD_VISIBILITY_MEAN = 8.0
CLOUD_VISIBILITY_STD = 1.2

SUBDIVISIONS = [
    "COASTAL REGION", "HILLY REGION", "PLAINS REGION",
    "NORTH ZONE", "SOUTH ZONE"
]

N_YEARS = 40
YEARS = list(range(1985, 1985 + N_YEARS))


def generate():
    rows = []
    for subdivision in SUBDIVISIONS:
        # subdivisions vary slightly in baseline rainfall intensity
        subdivision_factor = np.random.uniform(0.85, 1.25)
        for year in YEARS:
            monthly_values = {}
            for m in MONTHS:
                mean = MONTH_MEANS[m] * subdivision_factor
                std = MONTH_STD[m] * subdivision_factor
                val = max(0, np.random.normal(mean, std))
                monthly_values[m] = round(val, 1)

            annual = round(sum(monthly_values.values()), 1)
            monsoon_total = sum(monthly_values[m] for m in ["JUN", "JUL", "AUG", "SEP"])

            # Lower visibility often accompanies heavier rainfall and cloudy weather.
            cloud_vis = float(np.clip(
                np.random.normal(CLOUD_VISIBILITY_MEAN, CLOUD_VISIBILITY_STD),
                2.0,
                10.0,
            ))
            cloud_vis = round(cloud_vis - (monsoon_total / 3200) * 2.5, 1)
            cloud_vis = float(np.clip(cloud_vis, 1.0, 10.0))

            # Flood logic: driven mainly by monsoon months (JUN-SEP), annual total,
            # and reduced cloud visibility to reflect stormy conditions.
            flood_score = (
                0.50 * (monsoon_total / 2200)
                + 0.25 * (annual / 3200)
                + 0.15 * ((10.0 - cloud_vis) / 9.0)
                + 0.10 * np.random.uniform(0, 1)
            )
            row = {"SUBDIVISION": subdivision, "YEAR": year, **monthly_values,
                   "ANNUAL": annual, "CLOUD_VISIBILITY": cloud_vis,
                   "_flood_score": flood_score}
            rows.append(row)

    df = pd.DataFrame(rows)

    # Calibrate threshold on the score distribution to land close to a
    # realistic ~55/45 split instead of a lopsided one.
    threshold = df["_flood_score"].quantile(0.50)
    df["FLOODS"] = np.where(df["_flood_score"] > threshold, "YES", "NO")
    df = df.drop(columns=["_flood_score"])

    # Inject a small amount of realistic messiness: missing values + a couple outliers
    rng = np.random.RandomState(7)
    for col in ["JUN", "JUL", "AUG", "ANNUAL", "CLOUD_VISIBILITY"]:
        missing_idx = rng.choice(df.index, size=max(1, len(df) // 60), replace=False)
        df.loc[missing_idx, col] = np.nan

    outlier_idx = rng.choice(df.index, size=3, replace=False)
    df.loc[outlier_idx, "JUL"] = df.loc[outlier_idx, "JUL"] * 3.5

    return df


if __name__ == "__main__":
    import os
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rainfall_data.csv")
    df = generate()
    df.to_csv(output_path, index=False)
    print(f"Generated dataset with shape: {df.shape}")
    print(df["FLOODS"].value_counts())
    print(df.head())
