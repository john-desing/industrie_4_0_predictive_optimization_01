import pandas as pd
import os
import sys

# Ajouter le répertoire parent au chemin pour l'importation du module database_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database_manager import get_config_value # Importez la nouvelle fonction

def analyze_quality_data(df):
    """
    Analyse les données de qualité d'un DataFrame et détermine si le produit est hors spécifications.
    Les seuils sont maintenant récupérés dynamiquement depuis la base de données.
    """
    df_analyzed = df.copy()

    # Récupérer les seuils depuis la base de données
    ph_min = get_config_value('ph_product_min_threshold')
    ph_max = get_config_value('ph_product_max_threshold')
    color_min = get_config_value('color_intensity_min_threshold')
    color_max = get_config_value('color_intensity_max_threshold')
    dim_x_min = get_config_value('dimension_x_min_threshold')
    dim_x_max = get_config_value('dimension_x_max_threshold')

    # Vérification des spécifications
    df_analyzed['ph_out_of_spec'] = ((df_analyzed['ph_product'] < ph_min) | (df_analyzed['ph_product'] > ph_max)).astype(int)
    df_analyzed['color_out_of_spec'] = ((df_analyzed['color_intensity'] < color_min) | (df_analyzed['color_intensity'] > color_max)).astype(int)
    df_analyzed['dimension_x_out_of_spec'] = ((df_analyzed['dimension_x'] < dim_x_min) | (df_analyzed['dimension_x'] > dim_x_max)).astype(int)

    # Une alerte qualité est émise si au moins un paramètre est hors spécifications
    df_analyzed['quality_alert'] = ((df_analyzed['ph_out_of_spec'] == 1) | 
                                     (df_analyzed['color_out_of_spec'] == 1) | 
                                     (df_analyzed['dimension_x_out_of_spec'] == 1)).astype(int)
    
    return df_analyzed

if __name__ == "__main__":
    # Test simple avec des données simulées
    from data_simulator import generate_sensor_data_df
    from database_manager import create_tables, insert_sensor_data # Assurez-vous que la DB est prête

    create_tables() # S'assure que les tables et les seuils par défaut existent

    # Générer quelques données
    df_test = generate_sensor_data_df(num_records=10)
    insert_sensor_data(df_test) # Optionnel: insérer pour voir les seuils en action sur la DB si besoin

    # Analyser la qualité
    df_result = analyze_quality_data(df_test)
    print("Résultats de l'analyse qualité:")
    print(df_result[['timestamp', 'ph_product', 'color_intensity', 'dimension_x', 'quality_alert', 'ph_out_of_spec', 'color_out_of_spec', 'dimension_x_out_of_spec']])

    # Test avec une valeur hors seuil intentionnelle
    df_test_alert = generate_sensor_data_df(num_records=1)
    df_test_alert.loc[0, 'ph_product'] = 5.0 # Force une valeur hors seuil
    df_result_alert = analyze_quality_data(df_test_alert)
    print("\nRésultat d'une alerte forcée:")
    print(df_result_alert[['timestamp', 'ph_product', 'quality_alert', 'ph_out_of_spec']])