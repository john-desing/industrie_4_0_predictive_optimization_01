import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database_manager import (  # noqa: E402
    create_tables,
    fetch_sensor_data,
    insert_maintenance_prediction,
)


MODEL_NAME = "RandomForestPredictiveMaintenance"
EXPERIMENT_NAME = "Predictive Maintenance - Industrie 4.0"
FEATURE_COLUMNS = [
    "temperature_motor",
    "vibration_motor",
    "pressure_oil",
    "temp_motor_lag1",
    "vib_motor_lag1",
]
class ModelNotFoundError(RuntimeError):
    """Raised when the MLflow registry has no version for the requested model."""


def build_maintenance_features(data, previous_reading=None):
    """Build lag features from actual sensor observations in timestamp order."""
    current = data.copy().sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    current["_is_current_reading"] = True

    if previous_reading is not None:
        previous = pd.DataFrame([previous_reading])
        previous["_is_current_reading"] = False
        combined = pd.concat([previous, current], ignore_index=True, sort=False)
        combined = combined.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    else:
        combined = current

    combined["temp_motor_lag1"] = combined["temperature_motor"].shift(1)
    combined["vib_motor_lag1"] = combined["vibration_motor"].shift(1)
    features = combined.loc[
        combined["_is_current_reading"] & combined["temp_motor_lag1"].notna()
    ].copy()
    return features.drop(columns="_is_current_reading").reset_index(drop=True)


def load_and_preprocess_data():
    """Load sensor readings, label them, and create chronological lag features."""
    create_tables()
    sensor_data = fetch_sensor_data()
    if sensor_data.empty:
        raise RuntimeError(
            "Aucune donnée de capteur trouvée. Générez des données avant l'entraînement."
        )

    sensor_data["failure"] = (
        (sensor_data["temperature_motor"] > 90)
        | (sensor_data["vibration_motor"] > 3.5)
        | (sensor_data["pressure_oil"] < 2.0)
    ).astype(int)
    maintenance_data = build_maintenance_features(sensor_data)
    features = maintenance_data[FEATURE_COLUMNS]
    labels = maintenance_data["failure"]
    row_info = maintenance_data[["id", "timestamp"]]
    return features, labels, row_info


def train_predictive_model(
    features,
    labels,
    n_estimators=100,
    max_depth=None,
    model_name=MODEL_NAME,
):
    """Fit preprocessing on the chronological training split and register the pipeline."""
    if len(features) < 5:
        raise ValueError("Au moins cinq observations valides sont nécessaires.")

    features = features.loc[:, FEATURE_COLUMNS].reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    split_index = int(len(features) * 0.7)
    if split_index < 2 or split_index >= len(features):
        raise ValueError("Les jeux d'entraînement et de test doivent être non vides.")

    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = labels.iloc[:split_index], labels.iloc[split_index:]
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="predictive_maintenance_model",
            registered_model_name=model_name,
            input_example=X_train.head(5),
        )

        print("\n--- Rapport de Classification ---")
        print(classification_report(y_test, predictions, zero_division=0))
        print("\n--- Matrice de Confusion ---")
        print(confusion_matrix(y_test, predictions))
        print(f"Modèle enregistré dans MLflow. Run ID: {run.info.run_id}")

    return model


def load_latest_registered_model(model_name=MODEL_NAME):
    """Load the newest registered sklearn pipeline, including its fitted scaler."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ModelNotFoundError(
            f"Aucune version du modèle '{model_name}' n'est enregistrée dans MLflow."
        )

    latest_version = max(versions, key=lambda version: int(version.version))
    model_uri = f"models:/{model_name}/{latest_version.version}"
    return mlflow.sklearn.load_model(model_uri)


def make_prediction(model, features):
    """Return binary predictions and class-1 probabilities for engineered features."""
    model_features = features.loc[:, FEATURE_COLUMNS]
    predictions = model.predict(model_features).astype(int)
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 1 in classes:
        probabilities = model.predict_proba(model_features)[:, classes.index(1)]
    else:
        probabilities = [0.0] * len(predictions)
    return predictions, probabilities


def make_prediction_and_store(model, data_to_predict_df, previous_reading=None):
    """Predict and persist rows with lag values taken from observed sensor readings."""
    features = build_maintenance_features(
        data_to_predict_df, previous_reading=previous_reading
    )
    if features.empty:
        print("Aucune observation avec un relevé précédent réel à prédire.")
        return

    predictions, probabilities = make_prediction(model, features)
    for index, row in features.iterrows():
        insert_maintenance_prediction(
            reading_id=row["id"],
            timestamp=row["timestamp"],
            prediction=int(predictions[index]),
            probability=float(probabilities[index]),
        )
    print(f"{len(features)} prédictions enregistrées dans la base de données.")


if __name__ == "__main__":
    features, labels, _ = load_and_preprocess_data()
    model = train_predictive_model(
        features, labels, n_estimators=120, max_depth=10
    )

    latest_sensor_data = fetch_sensor_data(limit=100)
    if not latest_sensor_data.empty:
        make_prediction_and_store(model, latest_sensor_data)
    else:
        print("Pas de nouvelles données pour prédire.")
