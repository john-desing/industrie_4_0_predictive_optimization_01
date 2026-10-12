"""Create a reproducible demo model when trained artifacts are unavailable."""

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


MODEL_PATH = "models/predictive_maintenance_model.pkl"
SCALER_PATH = "models/predictive_maintenance_scaler.pkl"


def main():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return

    rng = np.random.default_rng(42)
    current = rng.normal(loc=[70, 2.0, 3.0], scale=[12, 1.0, 0.8], size=(5000, 3))
    previous = current + rng.normal(loc=0, scale=[4, 0.4, 0.2], size=(5000, 3))
    features = np.column_stack((current, previous[:, :2]))
    labels = (
        (features[:, 0] > 90)
        | (features[:, 1] > 3.5)
        | (features[:, 2] < 2.0)
    ).astype(int)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(scaled_features, labels)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


if __name__ == "__main__":
    main()
