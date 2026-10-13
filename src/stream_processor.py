import pandas as pd
import time
import os
import sys
import datetime
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_simulator import generate_sensor_data_df
from src.quality_control import analyze_quality_data
from src.database_manager import (
    DATABASE_PATH,
    create_tables,
    fetch_sensor_data,
    get_config_value,
    insert_maintenance_prediction,
    insert_quality_alert,
    insert_sensor_data,
)
from src.alert_manager import log_alert
from src.predictive_maintenance import (
    build_maintenance_features,
    load_latest_registered_model,
    make_prediction,
)


def load_model_from_mlflow(model_name="RandomForestPredictiveMaintenance"):
    """Load the latest registered pipeline, including its fitted preprocessing."""
    try:
        return load_latest_registered_model(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger le modèle '{model_name}' depuis MLflow."
        ) from e

def make_prediction_mlflow(model, new_data_dict, previous_reading):
    """Predict one reading using the actual immediately preceding sensor reading."""
    features = build_maintenance_features(
        pd.DataFrame([new_data_dict]), previous_reading=previous_reading
    )
    if features.empty:
        raise ValueError("Aucun relevé précédent réel n'est disponible pour le lag.")
    predictions, probabilities = make_prediction(model, features)
    return int(predictions[0]), float(probabilities[0])


if __name__ == "__main__":
    create_tables() # S'assure que les tables existent au démarrage du processeur
    model = load_model_from_mlflow()
    
    print("\n--- Démarrage de la simulation de flux de données avec modèle MLflow et DB ---")
    print("Appuyez sur Ctrl+C pour arrêter.")

    try:
        i = 0
        while True:
            # Récupérer les seuils dynamiquement
            ml_prob_threshold = get_config_value('ml_failure_prob_threshold')

            previous_readings = fetch_sensor_data(limit=1)
            if previous_readings.empty:
                raise RuntimeError(
                    "Générez au moins un relevé capteur avant de lancer le processeur."
                )
            previous_reading = previous_readings.iloc[0].to_dict()
            current_timestamp = datetime.datetime.now()
            single_record_df = generate_sensor_data_df(num_records=1, start_date=current_timestamp, introduce_drift=True)
            single_record_dict = single_record_df.iloc[0].to_dict()
            
            insert_sensor_data(single_record_df)
            
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sensor_readings ORDER BY id DESC LIMIT 1")
            reading_id = cursor.fetchone()[0]
            conn.close()

            prediction_maintenance, probability_maintenance = make_prediction_mlflow(
                model, single_record_dict, previous_reading
            )
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