import pandas as pd
import numpy as np
import datetime
import random
import os
import sys

# Ajouter le répertoire parent au chemin pour l'importation du module database_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database_manager import create_tables, insert_sensor_data

def generate_sensor_data_df(num_records=1000, start_date=None, introduce_drift=False):
    """
    Génère des données simulées pour divers capteurs industriels sous forme de DataFrame.
    Peut introduire une dérive de données après un certain point.
    """
    if start_date is None:
        start_date = datetime.datetime.now()

    data = []
    
    for i in range(num_records):
        timestamp = start_date + datetime.timedelta(minutes=i)
        
        # Base parameters
        temp_base = 75
        vib_base = 1.5
        pressure_base = 2.5
        ph_base = 7.0
        color_base = 150
        dim_base = 10.2

        # Introduce drift after a certain number of records or based on a flag
        drift_factor_temp = 0
        drift_factor_vib = 0
        drift_factor_ph = 0
        if introduce_drift and i > num_records * 0.5: # Drift starts after half of the records
            drift_factor_temp = 5 # Température augmente de 5 degrés en moyenne
            drift_factor_vib = 0.5 # Vibration augmente de 0.5 en moyenne
            drift_factor_ph = -0.3 # pH diminue de 0.3 en moyenne

        # Données pour la maintenance prédictive (ex: machine 1 - Moteur)
        temperature_motor = round(random.uniform(60, 95) + np.sin(i / 100) * 10 + np.random.normal(0, 2) + drift_factor_temp, 2)
        vibration_motor = round(random.uniform(0.5, 3.0) + np.cos(i / 50) * 0.8 + np.random.normal(0, 0.1) + drift_factor_vib, 2)
        pressure_oil = round(random.uniform(1.5, 4.0) + np.sin(i / 70) * 0.5 + np.random.normal(0, 0.05), 2)
        
        # Simuler une défaillance potentielle (valeurs anormales)
        if i > num_records * 0.8 and i % 5 == 0:
            temperature_motor += random.uniform(10, 20)
            vibration_motor += random.uniform(1.0, 2.0)
            pressure_oil -= random.uniform(0.5, 1.0)

        # Données pour le contrôle qualité (ex: Produit A)
        ph_product = round(random.uniform(6.5, 7.5) + np.random.normal(0, 0.1) + drift_factor_ph, 2)
        color_intensity = round(random.uniform(100, 200) + np.random.normal(0, 5), 2)
        dimension_x = round(random.uniform(10.0, 10.5) + np.random.normal(0, 0.05), 2)
        
        # Simuler un défaut qualité (valeurs hors tolérance)
        if i > num_records * 0.7 and i % 10 == 0:
            ph_product = round(random.uniform(5.0, 6.0), 2) # pH trop bas
            
        data.append([
            timestamp,
            temperature_motor,
            vibration_motor,
            pressure_oil,
            ph_product,
            color_intensity,
            dimension_x
        ])

    df = pd.DataFrame(data, columns=[
        'timestamp', 
        'temperature_motor', 'vibration_motor', 'pressure_oil',
        'ph_product', 'color_intensity', 'dimension_x'
    ])
    
    return df

def generate_and_store_sensor_data(num_records=1000, start_date=None, introduce_drift=False):
    """
    Génère des données de capteurs et les insère dans la base de données.
    """
    create_tables() # S'assure que les tables existent
    df_simulated = generate_sensor_data_df(num_records, start_date, introduce_drift)
    insert_sensor_data(df_simulated)
    print(f"Génération et stockage de {len(df_simulated)} enregistrements de capteurs terminés dans la base de données.")
    return df_simulated # Retourne le DataFrame aussi pour usage immédiat

if __name__ == "__main__":
    # Exemple: générer des données avec dérive pour un test local
    generate_and_store_sensor_data(num_records=5000, introduce_drift=True)
    print("Données avec dérive générées pour le test.")