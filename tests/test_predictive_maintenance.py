import tempfile
import unittest
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.predictive_maintenance import (
    FEATURE_COLUMNS,
    MODEL_NAME,
    ModelNotFoundError,
    build_maintenance_features,
    load_latest_registered_model,
    make_prediction,
    train_predictive_model,
)


def make_sensor_data(count=1000):
    rng = np.random.default_rng(7)
    values = rng.normal(
        loc=[70, 2.0, 3.0],
        scale=[12, 1.0, 0.8],
        size=(count, 3),
    )
    data = pd.DataFrame(
        values,
        columns=["temperature_motor", "vibration_motor", "pressure_oil"],
    )
    data["timestamp"] = pd.date_range("2024-01-01", periods=count, freq="min")
    data["failure"] = (
        (data["temperature_motor"] > 90)
        | (data["vibration_motor"] > 3.5)
        | (data["pressure_oil"] < 2.0)
    ).astype(int)
    return data


class PredictiveMaintenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        mlflow.set_registry_uri(None)
        mlflow.set_tracking_uri("sqlite://")
        cls.model_name = f"{MODEL_NAME}-test"
        artifact_root = Path(cls.temp_dir.name, "artifacts")
        artifact_root.mkdir()
        mlflow.create_experiment(
            "Predictive Maintenance - Industrie 4.0",
            artifact_location=artifact_root.as_uri(),
        )

    @classmethod
    def tearDownClass(cls):
        mlflow.end_run()
        mlflow.set_registry_uri(None)
        mlflow.set_tracking_uri("file:///tmp/mlflow-test-cleanup")
        cls.temp_dir.cleanup()

    def test_missing_registry_model_is_reported(self):
        with self.assertRaises(ModelNotFoundError):
            load_latest_registered_model(self.model_name)

    def test_training_registers_reloadable_pipeline_and_predicts(self):
        sensor_data = make_sensor_data()
        sensor_data.loc[700:, ["temperature_motor", "vibration_motor", "pressure_oil"]] += 500
        maintenance_data = build_maintenance_features(sensor_data)
        features = maintenance_data[FEATURE_COLUMNS]
        labels = maintenance_data["failure"]
        split_index = int(len(features) * 0.7)

        trained_model = train_predictive_model(
            features,
            labels,
            n_estimators=25,
            max_depth=6,
            model_name=self.model_name,
        )
        expected_training_means = features.iloc[:split_index].mean().to_numpy()
        actual_training_means = (
            trained_model.named_steps["scaler"].mean_
        )
        np.testing.assert_allclose(actual_training_means, expected_training_means)
        self.assertFalse(np.allclose(actual_training_means, features.mean().to_numpy()))

        loaded_model = load_latest_registered_model(self.model_name)
        predictions, probabilities = make_prediction(
            loaded_model, features.iloc[-10:]
        )
        self.assertEqual(len(predictions), 10)
        self.assertEqual(len(probabilities), 10)
        self.assertTrue(set(predictions).issubset({0, 1}))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_inference_lags_use_the_previous_observation(self):
        readings = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-01 00:00:00"),
                    "temperature_motor": 60.0,
                    "vibration_motor": 1.0,
                    "pressure_oil": 3.0,
                }
            ]
        )
        previous_reading = {
            "timestamp": pd.Timestamp("2023-12-31 23:59:00"),
            "temperature_motor": 42.5,
            "vibration_motor": 0.75,
            "pressure_oil": 3.2,
        }

        features = build_maintenance_features(
            readings, previous_reading=previous_reading
        )

        self.assertEqual(features.loc[0, "temp_motor_lag1"], 42.5)
        self.assertEqual(features.loc[0, "vib_motor_lag1"], 0.75)
        self.assertEqual(features.loc[0, "temperature_motor"], 60.0)


if __name__ == "__main__":
    unittest.main()
