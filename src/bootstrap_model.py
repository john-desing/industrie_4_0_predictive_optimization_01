"""Register a deterministic demo model when the MLflow registry is empty."""

import time
import os
import sys

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictive_maintenance import (
    MODEL_NAME,
    build_maintenance_features,
    train_predictive_model,
)


def _demo_training_data():
    rng = np.random.default_rng(42)
    sensor_values = rng.normal(
        loc=[70, 2.0, 3.0],
        scale=[12, 1.0, 0.8],
        size=(1500, 3),
    )
    data = pd.DataFrame(
        sensor_values,
        columns=["temperature_motor", "vibration_motor", "pressure_oil"],
    )
    data["timestamp"] = pd.date_range("2024-01-01", periods=len(data), freq="min")
    data["failure"] = (
        (data["temperature_motor"] > 90)
        | (data["vibration_motor"] > 3.5)
        | (data["pressure_oil"] < 2.0)
    ).astype(int)
    features = build_maintenance_features(data)
    return features, features["failure"]


def ensure_demo_model_registered():
    client = MlflowClient()
    for attempt in range(30):
        try:
            if client.search_model_versions(f"name='{MODEL_NAME}'"):
                print(f"Le modèle '{MODEL_NAME}' existe déjà dans le registre MLflow.")
                return
            break
        except MlflowException as error:
            if attempt == 29:
                raise RuntimeError(
                    "Le serveur MLflow n'est pas disponible après 60 secondes."
                ) from error
            print("En attente du serveur MLflow...")
            time.sleep(2)

    features, labels = _demo_training_data()
    train_predictive_model(
        features,
        labels,
        n_estimators=120,
        max_depth=10,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    ensure_demo_model_registered()
