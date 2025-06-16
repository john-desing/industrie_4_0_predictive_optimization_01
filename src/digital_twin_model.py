# Dans src/digital_twin_model.py (nouveau fichier)
import pandas as pd
import mlflow
import os

# Assurez-vous que MLFLOW_TRACKING_URI pointe vers votre serveur Docker
# Cela est crucial si vous exécutez ce script localement.
# Pour un environnement Dockerisé, le service Streamlit aurait accès à http://mlflow_server:5000
# Si exécuté localement, décommenter la ligne suivante:
# os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

# Récupérer les modèles depuis MLflow Registry
# Assurez-vous que les noms des modèles et les étapes sont corrects
model_maintenance = mlflow.pyfunc.load_model("models:/MaintenancePredictionModel/Production")
model_quality = mlflow.pyfunc.load_model("models:/QualityControlModel/Production")

def simulate_machine_state(temp_motor, vib_motor, pressure_oil, ph_product, color_intensity, dimension_x, num_hours_running):
    """
    Simule l'état d'une machine et prédit les risques de défaillance et de qualité.
    Prend des valeurs hypothétiques pour les capteurs et les utilise avec les modèles ML.
    """
    # Crée un DataFrame avec les caractéristiques dans le bon ordre pour le modèle
    # Assurez-vous que l'ordre des colonnes correspond à celui attendu par vos modèles entraînés
    input_data_maintenance = pd.DataFrame([{
        'temperature_motor': temp_motor,
        'vibration_motor': vib_motor,
        'pressure_oil': pressure_oil,
        'num_hours_running': num_hours_running
    }])

    input_data_quality = pd.DataFrame([{
        'ph_product': ph_product,
        'color_intensity': color_intensity,
        'dimension_x': dimension_x
    }])

    # Prédictions du modèle de maintenance
    maintenance_predictions = model_maintenance.predict(input_data_maintenance)
    maintenance_prob = maintenance_predictions[0] # Assurez-vous que c'est la probabilité de défaillance (si votre modèle de maintenance renvoie une probabilité)
    # S'il renvoie directement 0/1, ajustez ici

    # Prédictions du modèle de qualité
    quality_predictions = model_quality.predict(input_data_quality)
    quality_alert = quality_predictions[0] # Supposons 0=OK, 1=Alerte

    return {
        "prediction_maintenance_prob": maintenance_prob,
        "prediction_quality_alert": quality_alert
    }

# Exemple d'utilisation (pour test)
if __name__ == "__main__":
    # Assurez-vous que le serveur MLflow est démarré et que les modèles sont enregistrés dans le registre
    # mlflow_server doit être accessible (ex: http://localhost:5000 si exécuté localement)
    sim_result = simulate_machine_state(
        temp_motor=95, vib_motor=4.0, pressure_oil=1.8,
        ph_product=7.6, color_intensity=210, dimension_x=10.7,
        num_hours_running=500
    )
    print("Résultat de la simulation:")
    print(f"Probabilité de défaillance (maintenance): {sim_result['prediction_maintenance_prob']:.2f}")
    print(f"Alerte qualité: {'Oui' if sim_result['prediction_quality_alert'] == 1 else 'Non'}")