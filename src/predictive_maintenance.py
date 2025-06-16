import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
import os
import sys

# Ajouter le répertoire parent au chemin pour l'importation du module database_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database_manager import fetch_sensor_data, insert_maintenance_prediction, create_tables

# --- MLflow Configuration ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Predictive Maintenance - Industrie 4.0")

def load_and_preprocess_data():
    """
    Charge les données des capteurs depuis la DB et les prétraite pour la maintenance prédictive.
    """
    create_tables()
    df = fetch_sensor_data()
    if df.empty:
        print("Aucune donnée de capteur trouvée dans la base de données. Veuillez exécuter data_simulator.py en premier.")
        sys.exit(1)

    df_maintenance = df[['id', 'timestamp', 'temperature_motor', 'vibration_motor', 'pressure_oil']].copy()

    # Étiquetage simplifié
    df_maintenance['failure'] = ((df_maintenance['temperature_motor'] > 90) |
                                 (df_maintenance['vibration_motor'] > 3.5) |
                                 (df_maintenance['pressure_oil'] < 2.0)).astype(int)

    # Trier par timestamp
    df_maintenance = df_maintenance.sort_values('timestamp').reset_index(drop=True)
    df_maintenance['temp_motor_lag1'] = df_maintenance['temperature_motor'].shift(1)
    df_maintenance['vib_motor_lag1'] = df_maintenance['vibration_motor'].shift(1)
    df_maintenance.dropna(inplace=True)

    X = df_maintenance[['temperature_motor', 'vibration_motor', 'pressure_oil', 'temp_motor_lag1', 'vib_motor_lag1']]
    y = df_maintenance['failure']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, df_maintenance[['id', 'timestamp']]

def train_predictive_model(X, y, n_estimators=100, max_depth=None):
    """
    Entraîne un modèle de classification pour la maintenance prédictive
    et logue l'expérimentation avec MLflow.
    """
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print("\n--- Rapport de Classification ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\n--- Matrice de Confusion ---")
        print(confusion_matrix(y_test, y_pred))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="predictive_maintenance_model",
            registered_model_name="RandomForestPredictiveMaintenance"
        )
        print(f"Modèle loggé dans MLflow avec Run ID: {run.info.run_id}")
        
        return model

def make_prediction_and_store(model, scaler, data_to_predict_df):
    """
    Fait des prédictions sur de nouvelles données et les stocke dans la base de données.
    """
    if data_to_predict_df.empty:
        print("Aucune donnée à prédire.")
        return

    data_to_predict_df = data_to_predict_df.sort_values('timestamp').reset_index(drop=True)
    data_to_predict_df['temp_motor_lag1'] = data_to_predict_df['temperature_motor'].shift(1)
    data_to_predict_df['vib_motor_lag1'] = data_to_predict_df['vibration_motor'].shift(1)
    data_to_predict_df.dropna(inplace=True)

    features = data_to_predict_df[['temperature_motor', 'vibration_motor', 'pressure_oil', 'temp_motor_lag1', 'vib_motor_lag1']]
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[:, 1]

    for idx, row in data_to_predict_df.iterrows():
        insert_maintenance_prediction(
            reading_id=row['id'],
            timestamp=row['timestamp'],
            prediction=int(prediction[data_to_predict_df.index.get_loc(idx)]),
            probability=float(probability[data_to_predict_df.index.get_loc(idx)])
        )
    print(f"{len(data_to_predict_df)} prédictions enregistrées dans la base de données.")

# --------------------------------------------------
# 🔁 Fonctions utiles pour app.py (ajouts)
# --------------------------------------------------

def preprocess_maintenance_data(df_raw, scaler):
    """
    Prétraite de nouvelles données pour la prédiction.
    """
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
    df_raw['temp_motor_lag1'] = df_raw['temperature_motor'].shift(1)
    df_raw['vib_motor_lag1'] = df_raw['vibration_motor'].shift(1)
    df_raw.dropna(inplace=True)

    features = df_raw[['temperature_motor', 'vibration_motor', 'pressure_oil', 'temp_motor_lag1', 'vib_motor_lag1']]
    features_scaled = scaler.transform(features)
    return df_raw, features_scaled

def make_prediction(model, scaled_features):
    """
    Retourne la prédiction et les probabilités.
    """
    predictions = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)[:, 1]
    return predictions, probabilities

# --------------------------------------------------
# 🚀 Lancement du pipeline
# --------------------------------------------------
if __name__ == "__main__":
    X, y, scaler, original_data_info = load_and_preprocess_data()
    
    model = train_predictive_model(X, y, n_estimators=120, max_depth=10)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, 'models/predictive_maintenance_model.pkl')
    joblib.dump(scaler, 'models/predictive_maintenance_scaler.pkl')
    print("Modèle et scaler sauvegardés localement dans le dossier 'models/'.")

    latest_sensor_data = fetch_sensor_data(limit=100)
    if not latest_sensor_data.empty:
        make_prediction_and_store(model, scaler, latest_sensor_data)
    else:
        print("Pas de nouvelles données pour prédire.")
