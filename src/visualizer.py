import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ajouter le répertoire parent au chemin pour l'importation des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database_manager import fetch_sensor_data, fetch_maintenance_predictions, fetch_quality_alerts, get_all_config_values

REPORTS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_sensor_trends(df_sensors, config):
    """Génère des graphiques des tendances des capteurs avec les seuils configurés."""
    if df_sensors.empty:
        print("Aucune donnée de capteur à visualiser.")
        return

    # Récupérer les seuils depuis la configuration
    temp_motor_alert_threshold = config.get('temp_motor_alert_threshold', 90.0)
    vib_motor_alert_threshold = config.get('vib_motor_alert_threshold', 3.5)
    pressure_oil_alert_threshold = config.get('pressure_oil_alert_threshold', 2.0)
    ph_product_min_threshold = config.get('ph_product_min_threshold', 6.5)
    ph_product_max_threshold = config.get('ph_product_max_threshold', 7.5)
    color_intensity_min_threshold = config.get('color_intensity_min_threshold', 100.0)
    color_intensity_max_threshold = config.get('color_intensity_max_threshold', 200.0)
    dimension_x_min_threshold = config.get('dimension_x_min_threshold', 10.0)
    dimension_x_max_threshold = config.get('dimension_x_max_threshold', 10.5)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))
    fig.suptitle('Tendances des Capteurs avec Seuils Configurés', fontsize=16)

    # Température Moteur
    sns.lineplot(x='timestamp', y='temperature_motor', data=df_sensors, ax=axes[0, 0])
    axes[0, 0].axhline(y=temp_motor_alert_threshold, color='r', linestyle='--', label=f'Seuil Alerte ({temp_motor_alert_threshold}°C)')
    axes[0, 0].set_title('Température Moteur')
    axes[0, 0].legend()

    # Vibration Moteur
    sns.lineplot(x='timestamp', y='vibration_motor', data=df_sensors, ax=axes[0, 1])
    axes[0, 1].axhline(y=vib_motor_alert_threshold, color='r', linestyle='--', label=f'Seuil Alerte ({vib_motor_alert_threshold} mm/s)')
    axes[0, 1].set_title('Vibration Moteur')
    axes[0, 1].legend()

    # Pression d'Huile
    sns.lineplot(x='timestamp', y='pressure_oil', data=df_sensors, ax=axes[1, 0])
    axes[1, 0].axhline(y=pressure_oil_alert_threshold, color='r', linestyle='--', label=f'Seuil Alerte ({pressure_oil_alert_threshold} bar)')
    axes[1, 0].set_title('Pression d\'Huile')
    axes[1, 0].legend()

    # pH Produit
    sns.lineplot(x='timestamp', y='ph_product', data=df_sensors, ax=axes[1, 1])
    axes[1, 1].axhline(y=ph_product_min_threshold, color='r', linestyle='--', label=f'Min pH ({ph_product_min_threshold})')
    axes[1, 1].axhline(y=ph_product_max_threshold, color='r', linestyle='--', label=f'Max pH ({ph_product_max_threshold})')
    axes[1, 1].set_title('pH Produit')
    axes[1, 1].legend()

    # Intensité de Couleur
    sns.lineplot(x='timestamp', y='color_intensity', data=df_sensors, ax=axes[2, 0])
    axes[2, 0].axhline(y=color_intensity_min_threshold, color='r', linestyle='--', label=f'Min Couleur ({color_intensity_min_threshold})')
    axes[2, 0].axhline(y=color_intensity_max_threshold, color='r', linestyle='--', label=f'Max Couleur ({color_intensity_max_threshold})')
    axes[2, 0].set_title('Intensité de Couleur')
    axes[2, 0].legend()

    # Dimension X
    sns.lineplot(x='timestamp', y='dimension_x', data=df_sensors, ax=axes[2, 1])
    axes[2, 1].axhline(y=dimension_x_min_threshold, color='r', linestyle='--', label=f'Min Dim X ({dimension_x_min_threshold})')
    axes[2, 1].axhline(y=dimension_x_max_threshold, color='r', linestyle='--', label=f'Max Dim X ({dimension_x_max_threshold})')
    axes[2, 1].set_title('Dimension X')
    axes[2, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORTS_DIR, 'sensor_trends_with_thresholds.png'))
    plt.close(fig)
    print(f"Graphique 'sensor_trends_with_thresholds.png' généré dans {REPORTS_DIR}")

def generate_prediction_summary(df_predictions, df_sensors, config):
    """Génère un graphique de résumé des prédictions de défaillance."""
    if df_predictions.empty or df_sensors.empty:
        print("Pas assez de données pour générer le résumé des prédictions.")
        return

    df_merged = pd.merge(df_sensors, df_predictions, left_on='id', right_on='reading_id', how='inner', suffixes=('_sensor', '_pred'))

    # Pourcentage de défaillances prédites
    total_records = len(df_merged)
    failure_predictions = df_merged[df_merged['prediction'] == 1]
    num_failures = len(failure_predictions)
    failure_percentage = (num_failures / total_records) * 100 if total_records > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    prediction_counts = df_merged['prediction'].value_counts().rename(index={0: 'Normal', 1: 'Risque de Défaillance'})
    sns.barplot(x=prediction_counts.index, y=prediction_counts.values, ax=ax, palette=['green', 'red'])
    ax.set_title(f'Répartition des Prédictions de Défaillance (Total: {total_records} | Risque: {failure_percentage:.2f}%)')
    ax.set_ylabel('Nombre d\'Occurrences')
    ax.set_xlabel('Statut')
    plt.savefig(os.path.join(REPORTS_DIR, 'prediction_summary.png'))
    plt.close(fig)
    print(f"Graphique 'prediction_summary.png' généré dans {REPORTS_DIR}")

    # Histogramme des probabilités de défaillance
    if not failure_predictions.empty:
        fig_prob, ax_prob = plt.subplots(figsize=(10, 6))
        sns.histplot(failure_predictions['probability'], bins=10, kde=True, ax=ax_prob, color='red')
        
        # Ajouter le seuil de probabilité ML configuré
        ml_prob_threshold = config.get('ml_failure_prob_threshold', 0.5)
        ax_prob.axvline(x=ml_prob_threshold, color='blue', linestyle='--', label=f'Seuil Alerte ML ({ml_prob_threshold})')
        
        ax_prob.set_title('Distribution des Probabilités pour les Risques de Défaillance')
        ax_prob.set_xlabel('Probabilité de Défaillance')
        ax_prob.set_ylabel('Fréquence')
        ax_prob.legend()
        plt.savefig(os.path.join(REPORTS_DIR, 'failure_probability_distribution.png'))
        plt.close(fig_prob)
        print(f"Graphique 'failure_probability_distribution.png' généré dans {REPORTS_DIR}")


def generate_quality_summary(df_quality_alerts, df_sensors, config):
    """Génère un graphique de résumé des alertes qualité."""
    if df_quality_alerts.empty or df_sensors.empty:
        print("Pas assez de données pour générer le résumé qualité.")
        return

    df_merged_quality = pd.merge(df_sensors, df_quality_alerts, left_on='id', right_on='reading_id', how='inner', suffixes=('_sensor', '_alert'))

    total_records = len(df_merged_quality)
    alerts_count = df_merged_quality['quality_alert'].sum()
    alert_percentage = (alerts_count / total_records) * 100 if total_records > 0 else 0

    # Résumé des alertes par type
    quality_alert_counts = df_merged_quality[['ph_out_of_spec', 'color_out_of_spec', 'dimension_x_out_of_spec']].sum()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Résumé des Alertes Qualité (Total: {total_records} | Alertes: {alert_percentage:.2f}%)', fontsize=16)

    # Barplot des types d'alertes
    sns.barplot(x=quality_alert_counts.index, y=quality_alert_counts.values, palette='viridis', ax=axes[0])
    axes[0].set_title('Nombre d\'Alertes Qualité par Type')
    axes[0].set_ylabel('Nombre d\'Alertes')
    axes[0].set_xticklabels(['pH Hors Spéc.', 'Couleur Hors Spéc.', 'Dimension X Hors Spéc.'])

    # Pie chart de la répartition Conformité/Alerte
    labels = ['Conforme', 'Alerte Qualité']
    sizes = [total_records - alerts_count, alerts_count]
    colors = ['lightgreen', 'salmon']
    explode = (0, 0.1)  # explode 2nd slice
    axes[1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    axes[1].axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    axes[1].set_title('Répartition Conformité Produit')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORTS_DIR, 'quality_summary.png'))
    plt.close(fig)
    print(f"Graphique 'quality_summary.png' généré dans {REPORTS_DIR}")

def generate_all_reports():
    """Génère tous les rapports statiques."""
    print("Génération des rapports statiques...")
    df_sensors = fetch_sensor_data(limit=5000)
    df_predictions = fetch_maintenance_predictions(limit=5000)
    df_quality_alerts = fetch_quality_alerts(limit=5000)
    current_config = get_all_config_values() # Récupère les seuils actuels

    generate_sensor_trends(df_sensors, current_config)
    generate_prediction_summary(df_predictions, df_sensors, current_config)
    generate_quality_summary(df_quality_alerts, df_sensors, current_config)
    print("Tous les rapports statiques ont été générés.")

if __name__ == "__main__":
    generate_all_reports()