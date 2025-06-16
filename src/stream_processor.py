import pandas as pd
import time
import joblib
import os
import sys
import mlflow
import mlflow.pyfunc
import datetime
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_simulator import generate_sensor_data_df
from src.quality_control import analyze_quality_data
from src.database_manager import create_tables, insert_sensor_data, insert_maintenance_prediction, insert_quality_alert, DATABASE_PATH, get_config_value # Importez get_config_value
from src.alert_manager import log_alert

# --- MLflow Configuration (doit correspondre à la configuration d'entraînement) ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Predictive Maintenance - Industrie 4.0")
def load_models_from_mlflow(model_name="RandomForestPredictiveMaintenance"):
    """
    Charge le modèle le plus récent depuis le registre MLflow (sans dépendre du stage)
    et le scaler localement.
    """
    from mlflow.tracking import MlflowClient

    try:
        client = MlflowClient()
        latest_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not latest_versions:
            raise Exception(f"Aucune version du modèle '{model_name}' trouvée dans le registre MLflow.")
        
        # Trie par version décroissante pour obtenir la plus récente
        latest_version = max(latest_versions, key=lambda v: int(v.version))
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Modèle '{model_name}' version {latest_version.version} chargé depuis MLflow.")

        # Chargement du scaler
        scaler_path = os.path.join("models", "predictive_maintenance_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler non trouvé à l'emplacement : {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        print("Scaler de maintenance prédictive chargé localement.")

        return model, scaler

    except Exception as e:
        print(f"Erreur lors du chargement des modèles depuis MLflow ou localement : {e}")
        print("Assurez-vous que le modèle est enregistré dans MLflow et que le scaler est sauvegardé localement.")
        sys.exit(1)

def make_prediction_mlflow(model_mlflow_pyfunc, scaler, new_data_dict):
    """
    Fait une prédiction sur de nouvelles données en utilisant le modèle chargé via MLflow pyfunc.
    new_data_dict est un dictionnaire représentant une seule ligne de données de capteur.
    """
    temp_motor_lag1_val = new_data_dict['temperature_motor'] * 0.98
    vib_motor_lag1_val = new_data_dict['vibration_motor'] * 0.95

    features_for_prediction = pd.DataFrame([[
        new_data_dict['temperature_motor'],
        new_data_dict['vibration_motor'],
        new_data_dict['pressure_oil'],
        temp_motor_lag1_val,
        vib_motor_lag1_val
    ]], columns=['temperature_motor', 'vibration_motor', 'pressure_oil', 'temp_motor_lag1', 'vib_motor_lag1'])

    features_scaled = scaler.transform(features_for_prediction)
    prediction_array = model_mlflow_pyfunc.predict(features_scaled)
    prediction = prediction_array[0]
    
    probability = 0.0 
    if hasattr(model_mlflow_pyfunc.unwrap_python_model()._model, 'predict_proba'):
         probability = model_mlflow_pyfunc.unwrap_python_model()._model.predict_proba(features_scaled)[:, 1][0]
    else:
        probability = 0.9 if prediction == 1 else 0.1 # Simplification pour la démo

    return prediction, probability


if __name__ == "__main__":
    create_tables() # S'assure que les tables existent au démarrage du processeur
    model, scaler = load_models_from_mlflow()
    
    print("\n--- Démarrage de la simulation de flux de données avec modèle MLflow et DB ---")
    print("Appuyez sur Ctrl+C pour arrêter.")

    try:
        i = 0
        while True:
            # Récupérer les seuils dynamiquement
            ml_prob_threshold = get_config_value('ml_failure_prob_threshold')

            current_timestamp = datetime.datetime.now()
            single_record_df = generate_sensor_data_df(num_records=1, start_date=current_timestamp, introduce_drift=True)
            single_record_dict = single_record_df.iloc[0].to_dict()
            
            insert_sensor_data(single_record_df)
            
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sensor_readings ORDER BY id DESC LIMIT 1")
            reading_id = cursor.fetchone()[0]
            conn.close()

            prediction_maintenance, probability_maintenance = make_prediction_mlflow(model, scaler, single_record_dict)
            insert_maintenance_prediction(
                reading_id=reading_id,
                timestamp=single_record_dict['timestamp'],
                prediction=prediction_maintenance,
                probability=probability_maintenance
            )
            # Utiliser le seuil dynamique pour l'alerte
            if prediction_maintenance == 1 and probability_maintenance >= ml_prob_threshold: # Utilise le seuil configurable
                log_alert("Maintenance Prédictive", 
                          f"HAUTE PROBABILITÉ DE DÉFAILLANCE sur machine à {single_record_dict['timestamp']} (Prob: {probability_maintenance:.2f})", 
                          "CRITICAL")
            
            quality_df_single_record = analyze_quality_data(single_record_df) # analyze_quality_data lit déjà les seuils
            quality_alert_info = quality_df_single_record.iloc[0]
            insert_quality_alert(
                reading_id=reading_id,
                timestamp=quality_alert_info['timestamp'],
                quality_alert=int(quality_alert_info['quality_alert']),
                ph_out_of_spec=int(quality_alert_info['ph_out_of_spec']),
                color_out_of_spec=int(quality_alert_info['color_out_of_spec']),
                dimension_x_out_of_spec=int(quality_alert_info['dimension_x_out_of_spec'])
            )
            if quality_alert_info['quality_alert'] == 1:
                alert_details = []
                if quality_alert_info['ph_out_of_spec'] == 1: alert_details.append("pH")
                if quality_alert_info['color_out_of_spec'] == 1: alert_details.append("Couleur")
                if quality_alert_info['dimension_x_out_of_spec'] == 1: alert_details.append("Dimension X")
                
                log_alert("Contrôle Qualité", 
                          f"ALERTE QUALITÉ: Produit hors spécifications à {single_record_dict['timestamp']} (Problèmes: {', '.join(alert_details)})", 
                          "CRITICAL")
            
            print(f"[{single_record_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] - Machine: {'Défaillance Prédite' if prediction_maintenance == 1 else 'Normal'} (Prob: {probability_maintenance:.2f}) | Qualité Produit: {'Alerte Qualité' if quality_alert_info['quality_alert'] == 1 else 'Conforme'}")
            
            time.sleep(1)
            i += 1
            if i % 100 == 0:
                print(f"Processed {i} records...")
    except KeyboardInterrupt:
        print("\nSimulation de flux arrêtée.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue: {e}")