import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import datetime
import sqlite3
import plotly.graph_objects as go




# Ajouter le répertoire parent au chemin pour l'importation des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Importer les fonctions nécessaires de nos modules
from data_simulator import generate_sensor_data_df
from quality_control import analyze_quality_data 
from src.predictive_maintenance import (
    build_maintenance_features,
    load_latest_registered_model,
    make_prediction,
)

# Importer les modules de DB et d'alertes, y compris les nouvelles fonctions de config
from database_manager import create_tables, fetch_sensor_data, insert_sensor_data, \
                             fetch_maintenance_predictions, insert_maintenance_prediction, \
                             fetch_quality_alerts, insert_quality_alert, DATABASE_PATH, \
                             get_config_value, set_config_value, get_all_config_values, \
                             initialize_default_thresholds 

from alert_manager import log_alert, get_recent_alerts, ALERT_LOG_FILE

# Configuration de Streamlit
st.set_page_config(
    page_title="Industrie 4.0 : Optimisation Prédictive",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fonctions utilitaires de chargement ---
@st.cache_data(ttl=1) 
def get_sensor_data(limit=None):
    """ Récupère les données brutes des capteurs depuis la DB. """
    return fetch_sensor_data(limit=limit)

@st.cache_data(ttl=1)
def get_maintenance_predictions(limit=None):
    """ Récupère les prédictions de maintenance depuis la DB. """
    return fetch_maintenance_predictions(limit=limit)

@st.cache_data(ttl=1)
def get_quality_alerts(limit=None):
    """ Récupère les alertes qualité depuis la DB. """
    return fetch_quality_alerts(limit=limit)

@st.cache_data(ttl=1) 
def get_current_config():
    """ Récupère toutes les configurations actuelles. """
    return get_all_config_values()

@st.cache_resource
def load_maintenance_model():
    """Charge le modèle et son prétraitement depuis le registre MLflow."""
    return load_latest_registered_model()


model_maintenance = load_maintenance_model()

# Assurez-vous que la base de données et les tables existent, et que les seuils par défaut sont là
create_tables()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Vue d'ensemble", "Maintenance Prédictive", "Contrôle Qualité", "Surveillance de Dérive", "Simulation en Direct", "Historique des Alertes", "Configuration des Seuils","Jumeau Numériqu (Simulation)"]) 

st.sidebar.markdown("---")
st.sidebar.header("Gestion des Données")

if st.sidebar.button("Initialiser / Vider la Base de Données"):
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
    if os.path.exists(ALERT_LOG_FILE):
        os.remove(ALERT_LOG_FILE)
    st.cache_data.clear() 
    create_tables() 
    st.success("Base de données et journal d'alertes réinitialisés.")
    st.experimental_rerun()

num_records_to_simulate = st.sidebar.slider("Nombre d'enregistrements initiaux à générer", 100, 10000, 1000)
introduce_initial_drift = st.sidebar.checkbox("Introduire une dérive dans les données initiales", value=False)
if st.sidebar.button("Générer et stocker de nouvelles données"):
    generate_sensor_data_df(num_records=num_records_to_simulate, start_date=datetime.datetime.now(), introduce_drift=introduce_initial_drift)
    st.cache_data.clear() 
    st.success(f"{num_records_to_simulate} nouvelles données générées et stockées dans la DB (dérive: {introduce_initial_drift})!")
    st.experimental_rerun()

# --- PAGE: Vue d'ensemble ---
if page == "Vue d'ensemble":
    st.title("Tableau de Bord Industrie 4.0 : Vue d'ensemble")
    st.write("Bienvenue dans l'application d'optimisation prédictive pour l'industrie 4.0. Explorez les différentes sections pour comprendre la maintenance prédictive, le contrôle qualité et la surveillance du système.")

    df_sensors = get_sensor_data(limit=5000) # Augmenter la limite pour plus de données pour les KPIs
    df_predictions = get_maintenance_predictions(limit=5000)
    df_quality_alerts = get_quality_alerts(limit=5000)

    # --- KPIs Globaux ---
    st.subheader("Indicateurs Clés de Performance (KPIs)")
    if not df_sensors.empty:
        total_records = len(df_sensors)
        
        # KPI Maintenance Prédictive
        total_maintenance_predictions = len(df_predictions)
        predicted_failures = df_predictions[df_predictions['prediction'] == 1]
        num_predicted_failures = len(predicted_failures)
        failure_rate = (num_predicted_failures / total_maintenance_predictions * 100) if total_maintenance_predictions > 0 else 0

        # KPI Qualité Produit
        total_quality_alerts = len(df_quality_alerts)
        num_quality_issues = df_quality_alerts[df_quality_alerts['quality_alert'] == 1].shape[0]
        quality_defect_rate = (num_quality_issues / total_quality_alerts * 100) if total_quality_alerts > 0 else 0

        # KPI Dérive du Modèle (simple indication)
        # Pour une vraie dérive, il faudrait un suivi plus sophistiqué, ici on utilise le log des alertes
        recent_drift_alerts = [alert for alert in get_recent_alerts(num_alerts=20) if "Dérive du Modèle" in alert]
        drift_status = "Dérive Détectée Récemment" if recent_drift_alerts else "Pas de Dérive Récemment"
        drift_delta = f"-{len(recent_drift_alerts)} alertes" if recent_drift_alerts else "Stable"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Enregistrements Capteurs", value=total_records)
        with col2:
            st.metric(
                label="Taux de Risque de Défaillance", 
                value=f"{failure_rate:.2f}%", 
                delta=f"{num_predicted_failures} défaillances prédites", 
                delta_color="inverse" if failure_rate > 0 else "normal"
            )
        with col3:
            st.metric(
                label="Taux de Défaut Qualité", 
                value=f"{quality_defect_rate:.2f}%", 
                delta=f"{num_quality_issues} produits non-conformes", 
                delta_color="inverse" if quality_defect_rate > 0 else "normal"
            )
        with col4:
            st.metric(label="Statut de Dérive Modèle", value=drift_status, delta=drift_delta, delta_color="inverse" if recent_drift_alerts else "normal")

    else:
        st.info("Aucune donnée disponible pour les KPIs. Générez des données.")

    st.subheader("Aperçu des 1000 dernières données brutes de capteurs")
    if not df_sensors.empty:
        st.dataframe(df_sensors.head())
        st.write(f"Total d'enregistrements dans la DB: **{len(df_sensors)}** (affichés: {min(1000, len(df_sensors))})")

        st.subheader("Tendances récentes des principaux capteurs")
        cols = st.columns(3)
        with cols[0]:
            st.line_chart(df_sensors.set_index('timestamp')['temperature_motor'])
            st.write("Température Moteur")
        with cols[1]:
            st.line_chart(df_sensors.set_index('timestamp')['vibration_motor'])
            st.write("Vibration Moteur")
        with cols[2]:
            st.line_chart(df_sensors.set_index('timestamp')['ph_product'])
            st.write("pH Produit")
    else:
        st.info("Aucune donnée de capteur trouvée. Générez des données via la barre latérale ou démarrez la simulation en direct.")

    st.markdown("---")
    st.info("Utilisez la barre latérale pour naviguer entre les sections et gérer les données.")

# --- PAGE: Maintenance Prédictive ---
elif page == "Maintenance Prédictive":
    st.title("Maintenance Prédictive des Équipements")
    st.write("Analyse des données de capteurs pour prédire les défaillances des machines.")

    df_predictions = get_maintenance_predictions(limit=5000)
    df_sensors_full = get_sensor_data(limit=5000)

    if not df_predictions.empty and not df_sensors_full.empty:
        df_merged = pd.merge(df_sensors_full, df_predictions, left_on='id', right_on='reading_id', how='inner', suffixes=('_sensor', '_pred'))
        df_merged = df_merged.sort_values('timestamp_sensor', ascending=True) # Assurez-vous que c'est trié pour les graphiques

        st.subheader("Résumé des Prédictions de Défaillance")
        prediction_counts = df_merged['prediction'].value_counts().rename(index={0: 'Normal', 1: 'Risque de Défaillance'})
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(prediction_counts)
        with col2:
            st.metric(label="Pourcentage de défaillances prédites", value=f"{(prediction_counts.get('Risque de Défaillance', 0) / len(df_merged) * 100):.2f}%")

        # --- NOUVEAU: Histogramme des probabilités de défaillance ---
        st.subheader("Distribution des Probabilités de Défaillance")
        current_config = get_current_config()
        ml_prob_threshold = current_config.get('ml_failure_prob_threshold', 0.5)

        fig_prob, ax_prob = plt.subplots(figsize=(10, 6))
        sns.histplot(df_merged['probability'], bins=20, kde=True, ax=ax_prob, color='skyblue')
        ax_prob.axvline(x=ml_prob_threshold, color='red', linestyle='--', label=f'Seuil Alerte ML ({ml_prob_threshold:.2f})')
        ax_prob.set_title('Distribution des Probabilités de Défaillance Prédites')
        ax_prob.set_xlabel('Probabilité de Défaillance')
        ax_prob.set_ylabel('Fréquence')
        ax_prob.legend()
        st.pyplot(fig_prob)

        st.subheader("Tendances des Capteurs avec Alertes de Défaillance (100 dernières lectures)")
        
        # Récupérer les seuils depuis la DB pour l'affichage
        temp_motor_alert_threshold = current_config.get('temp_motor_alert_threshold', 90.0)
        vib_motor_alert_threshold = current_config.get('vib_motor_alert_threshold', 3.5)
        pressure_oil_alert_threshold = current_config.get('pressure_oil_alert_threshold', 2.0)


        fig_maint, ax_maint = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        df_display_maint = df_merged.tail(100) # Afficher les 100 dernières pour la clarté

        # Plot pour Température Moteur
        sns.lineplot(data=df_display_maint, x='timestamp_sensor', y='temperature_motor', hue='prediction', palette={0: 'blue', 1: 'red'}, ax=ax_maint[0], errorbar=None)
        ax_maint[0].set_title('Température Moteur et Prédiction de Défaillance')
        ax_maint[0].set_ylabel('Température (°C)')
        ax_maint[0].axhline(y=temp_motor_alert_threshold, color='grey', linestyle='--', alpha=0.7, label=f'Seuil Configuré ({temp_motor_alert_threshold})')
        ax_maint[0].legend()

        # Plot pour Vibration Moteur
        sns.lineplot(data=df_display_maint, x='timestamp_sensor', y='vibration_motor', hue='prediction', palette={0: 'blue', 1: 'red'}, ax=ax_maint[1], errorbar=None)
        ax_maint[1].set_title('Vibration Moteur et Prédiction de Défaillance')
        ax_maint[1].set_ylabel('Vibration (mm/s)')
        ax_maint[1].axhline(y=vib_motor_alert_threshold, color='grey', linestyle='--', alpha=0.7, label=f'Seuil Configuré ({vib_motor_alert_threshold})')
        ax_maint[1].legend()

        # Plot pour Pression d'Huile
        sns.lineplot(data=df_display_maint, x='timestamp_sensor', y='pressure_oil', hue='prediction', palette={0: 'blue', 1: 'red'}, ax=ax_maint[2], errorbar=None)
        ax_maint[2].set_title('Pression d\'Huile et Prédiction de Défaillance')
        ax_maint[2].set_ylabel('Pression (bar)')
        ax_maint[2].axhline(y=pressure_oil_alert_threshold, color='grey', linestyle='--', alpha=0.7, label=f'Seuil Configuré ({pressure_oil_alert_threshold})')
        ax_maint[2].legend()

        plt.tight_layout()
        st.pyplot(fig_maint)

        st.subheader("Détails des Enregistrements avec Risque de Défaillance")
        st.dataframe(df_merged[df_merged['prediction'] == 1].sort_values('probability', ascending=False))
    else:
        st.info("Aucune prédiction de maintenance trouvée. Exécutez `src/predictive_maintenance.py` et/ou générez des données via la simulation en direct.")


# --- PAGE: Contrôle Qualité ---
elif page == "Contrôle Qualité":
    st.title("Contrôle Qualité des Produits")
    st.write("Analyse des paramètres de production pour identifier les défauts de qualité.")

    df_quality_alerts = get_quality_alerts(limit=5000)
    df_sensors_full = get_sensor_data(limit=5000)

    if not df_quality_alerts.empty and not df_sensors_full.empty:
        df_merged_quality = pd.merge(df_sensors_full, df_quality_alerts, left_on='id', right_on='reading_id', how='inner', suffixes=('_sensor', '_alert'))
        df_merged_quality = df_merged_quality.sort_values('timestamp_sensor', ascending=True)

        st.subheader("Résumé des Alertes Qualité")
        total_records = len(df_merged_quality)
        alerts_count = df_merged_quality['quality_alert'].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Nombre total d'enregistrements analysés", value=total_records)
        with col2:
            st.metric(label="Nombre d'Alertes Qualité", value=alerts_count, delta=f"{(alerts_count / total_records * 100):.2f}% des produits non conformes")

        st.subheader("Distribution des Problèmes par Paramètre")
        quality_alert_counts = df_merged_quality[['ph_out_of_spec', 'color_out_of_spec', 'dimension_x_out_of_spec']].sum()
        
        fig_quality_bar, ax_quality_bar = plt.subplots(figsize=(10, 6))
        sns.barplot(x=quality_alert_counts.index, y=quality_alert_counts.values, palette='viridis', ax=ax_quality_bar)
        ax_quality_bar.set_title('Nombre d\'Alertes Qualité par Type')
        ax_quality_bar.set_ylabel('Nombre d\'Alertes')
        ax_quality_bar.set_xticklabels(['pH Hors Spéc.', 'Couleur Hors Spéc.', 'Dimension X Hors Spéc.'])
        st.pyplot(fig_quality_bar)

        # --- NOUVEAU: Visualisation des paramètres qualité avec les seuils dynamiques ---
        st.subheader("Tendances des Paramètres Qualité avec Seuils (100 dernières lectures)")
        current_config = get_current_config()
        ph_min = current_config.get('ph_product_min_threshold', 6.5)
        ph_max = current_config.get('ph_product_max_threshold', 7.5)
        color_min = current_config.get('color_intensity_min_threshold', 100.0)
        color_max = current_config.get('color_intensity_max_threshold', 200.0)
        dim_x_min = current_config.get('dimension_x_min_threshold', 10.0)
        dim_x_max = current_config.get('dimension_x_max_threshold', 10.5)

        fig_quality_trends, ax_quality_trends = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        df_display_quality = df_merged_quality.tail(100) # Afficher les 100 dernières pour la clarté

        # Plot pour pH Produit
        sns.lineplot(data=df_display_quality, x='timestamp_sensor', y='ph_product', hue='quality_alert', palette={0: 'blue', 1: 'red'}, ax=ax_quality_trends[0], errorbar=None)
        ax_quality_trends[0].set_title('pH Produit et Alerte Qualité')
        ax_quality_trends[0].set_ylabel('pH')
        ax_quality_trends[0].axhline(y=ph_min, color='grey', linestyle='--', alpha=0.7, label=f'Min pH ({ph_min})')
        ax_quality_trends[0].axhline(y=ph_max, color='grey', linestyle='--', alpha=0.7, label=f'Max pH ({ph_max})')
        ax_quality_trends[0].legend()

        # Plot pour Intensité de Couleur
        sns.lineplot(data=df_display_quality, x='timestamp_sensor', y='color_intensity', hue='quality_alert', palette={0: 'blue', 1: 'red'}, ax=ax_quality_trends[1], errorbar=None)
        ax_quality_trends[1].set_title('Intensité de Couleur et Alerte Qualité')
        ax_quality_trends[1].set_ylabel('Intensité')
        ax_quality_trends[1].axhline(y=color_min, color='grey', linestyle='--', alpha=0.7, label=f'Min Couleur ({color_min})')
        ax_quality_trends[1].axhline(y=color_max, color='grey', linestyle='--', alpha=0.7, label=f'Max Couleur ({color_max})')
        ax_quality_trends[1].legend()

        # Plot pour Dimension X
        sns.lineplot(data=df_display_quality, x='timestamp_sensor', y='dimension_x', hue='quality_alert', palette={0: 'blue', 1: 'red'}, ax=ax_quality_trends[2], errorbar=None)
        ax_quality_trends[2].set_title('Dimension X et Alerte Qualité')
        ax_quality_trends[2].set_ylabel('Dimension (mm)')
        ax_quality_trends[2].axhline(y=dim_x_min, color='grey', linestyle='--', alpha=0.7, label=f'Min Dim X ({dim_x_min})')
        ax_quality_trends[2].axhline(y=dim_x_max, color='grey', linestyle='--', alpha=0.7, label=f'Max Dim X ({dim_x_max})')
        ax_quality_trends[2].legend()

        plt.tight_layout()
        st.pyplot(fig_quality_trends)

        st.subheader("Détails des Alertes Qualité")
        st.dataframe(df_quality_alerts[df_quality_alerts['quality_alert'] == 1].sort_values('timestamp', ascending=False))
    else:
        st.info("Aucune alerte qualité trouvée. Exécutez `src/quality_control.py` et/ou générez des données via la simulation en direct.")

# --- PAGE: Surveillance de Dérive ---
elif page == "Surveillance de Dérive":
    st.title("Surveillance de la Dérive du Modèle (Model Drift)")
    st.write("Compare les statistiques des données entrantes récentes avec les données d'entraînement du modèle pour détecter une dérive potentielle.")

    # Chargement des données d'entraînement (simulées)
    # Utilisez une fonction pour charger ceci pour la reproductibilité
    @st.cache_data
    def get_training_reference_data():
        return generate_sensor_data_df(num_records=num_records_to_simulate, start_date=datetime.datetime(2023,1,1), introduce_drift=False)

    df_training_reference = get_training_reference_data()

    st.subheader("Paramètres surveillés pour la dérive")
    drift_features = ['temperature_motor', 'vibration_motor', 'ph_product', 'color_intensity', 'dimension_x'] # Plus de features pour la dérive
    
    # Récupérer le seuil de dérive dynamique
    drift_detection_percentage = get_current_config().get('drift_detection_percentage', 10.0)
    st.info(f"La dérive est détectée si la moyenne d'un capteur récent (dernières 100 lectures) s'écarte de plus de **{drift_detection_percentage:.2f}%** de sa moyenne d'entraînement (définie par les données initiales sans dérive).")

    df_recent_data = get_sensor_data(limit=100)
    
    if not df_recent_data.empty:
        drift_detected = False
        st.markdown("### Analyse de la Dérive :")
        
        # --- NOUVEAU: Visualisation de la dérive pour chaque feature ---
        fig_drift, axes_drift = plt.subplots(len(drift_features), 1, figsize=(15, 5 * len(drift_features)), sharex=True)
        if len(drift_features) == 1: # Gérer le cas où il n'y a qu'un seul sous-plot
            axes_drift = [axes_drift]

        for idx, feature in enumerate(drift_features):
            col_ref, col_recent, col_diff, col_status = st.columns(4)
            
            ref_mean = df_training_reference[feature].mean()
            recent_mean = df_recent_data[feature].mean()
            
            percentage_diff = abs((recent_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0

            status_icon = ""
            alert_level = ""
            if percentage_diff > drift_detection_percentage: 
                status_icon = "🚨 **DÉRIVE DÉTECTÉE**"
                drift_detected = True
                alert_level = "WARNING"
                log_alert("Dérive du Modèle", f"Dérive détectée sur '{feature}': Écart de {percentage_diff:.2f}% de la moyenne.", "WARNING")
            else:
                status_icon = "✅ Normal"
                alert_level = "NORMAL" # Pas un vrai niveau d'alerte mais pour la logique
            
            with col_ref:
                st.metric(label=f"Moyenne entraînement ({feature})", value=f"{ref_mean:.2f}")
            with col_recent:
                st.metric(label=f"Moyenne récente ({feature})", value=f"{recent_mean:.2f}")
            with col_diff:
                st.metric(label=f"Écart (%)", value=f"{percentage_diff:.2f}%")
            with col_status:
                st.markdown(f"**Statut :** {status_icon}")

            # Ajouter le graphique pour cette feature
            sns.lineplot(data=df_recent_data, x='timestamp', y=feature, ax=axes_drift[idx], color='blue', label='Données Récentes')
            axes_drift[idx].axhline(y=ref_mean, color='green', linestyle='--', label='Moyenne Entraînement')
            
            # Afficher la bande d'alerte basée sur le seuil de dérive
            upper_bound = ref_mean * (1 + drift_detection_percentage / 100)
            lower_bound = ref_mean * (1 - drift_detection_percentage / 100)
            axes_drift[idx].axhspan(lower_bound, upper_bound, color='red', alpha=0.1, label='Zone d\'Alerte Dérive')
            
            axes_drift[idx].set_title(f'Tendance de {feature} (récent vs entraînement)')
            axes_drift[idx].set_ylabel(feature)
            axes_drift[idx].legend()

        plt.tight_layout()
        st.pyplot(fig_drift)

        if drift_detected:
            st.warning("⚠️ **Attention : Une dérive de données potentielle a été détectée. Une réévaluation ou un réentraînement du modèle pourrait être nécessaire.**")
        else:
            st.success("👍 Aucune dérive significative détectée pour le moment.")
    else:
        st.info("Pas assez de données récentes pour évaluer la dérive. Lancez la simulation en direct.")

# --- PAGE: Simulation en Direct ---
elif page == "Simulation en Direct":
    st.title("Simulation de Flux de Données en Temps Réel")
    st.write("Observez les prédictions de maintenance et les alertes qualité en temps quasi réel pour des données générées à la volée. Les résultats sont stockés dans la base de données. **Une dérive des données est introduite dans ce flux.**")

    st.warning("Note: Cette simulation insère des données et des prédictions/alertes directement dans la base de données.")

    placeholder = st.empty()

    running_sim_flag = st.session_state.get('running_stream', False)

    col_sim_btn_1, col_sim_btn_2 = st.columns(2)
    with col_sim_btn_1:
        if st.button("Démarrer la simulation de flux", key="start_stream", disabled=running_sim_flag):
            st.session_state['running_stream'] = True
            st.session_state['stream_results_display'] = []
            st.write("Simulation en cours... (Appuyez sur 'Arrêter' pour stopper)")
            st.rerun()
    with col_sim_btn_2:
        if st.button("Arrêter la simulation de flux", key="stop_stream", disabled=not running_sim_flag):
            st.session_state['running_stream'] = False
            st.write("Simulation arrêtée.")
            st.rerun()

    if st.session_state.get('running_stream', False):
        try:
            i = 0
            while st.session_state['running_stream']:
                # Récupérer les seuils dynamiquement pour la simulation
                ml_prob_threshold = get_config_value('ml_failure_prob_threshold')

                current_timestamp = datetime.datetime.now()
                single_record_df = generate_sensor_data_df(num_records=1, start_date=current_timestamp, introduce_drift=True)
                single_record_dict = single_record_df.iloc[0].to_dict()
                previous_sensor_data = fetch_sensor_data(limit=1)
                if previous_sensor_data.empty:
                    st.error("Générez des données capteur avant de démarrer la simulation.")
                    st.session_state['running_stream'] = False
                    st.stop()
                previous_reading = previous_sensor_data.iloc[0].to_dict()
                
                insert_sensor_data(single_record_df)
                
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM sensor_readings ORDER BY id DESC LIMIT 1")
                reading_id = cursor.fetchone()[0]
                conn.close()

                features_for_prediction = build_maintenance_features(
                    single_record_df, previous_reading=previous_reading
                )
                predictions, probabilities = make_prediction(
                    model_maintenance, features_for_prediction
                )
                prediction_maintenance = int(predictions[0])
                probability_maintenance = float(probabilities[0])
                insert_maintenance_prediction(
                    reading_id=reading_id,
                    timestamp=single_record_dict['timestamp'],
                    prediction=prediction_maintenance,
                    probability=probability_maintenance
                )
                if prediction_maintenance == 1 and probability_maintenance >= ml_prob_threshold: 
                    log_alert("Maintenance Prédictive", 
                              f"HAUTE PROBABILITÉ DE DÉFAILLANCE sur machine à {single_record_dict['timestamp']} (Prob: {probability_maintenance:.2f})", 
                              "CRITICAL")
                
                # analyze_quality_data lit ses seuils de la DB
                quality_df_single_record = analyze_quality_data(single_record_df)
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
                
               # Déterminer l'état de la machine
                etat_machine = (
                    '<span style="color:red">Défaillance Prédite</span>' if prediction_maintenance == 1 else 'Normal'
                )

                # Déterminer l'état de la qualité produit
                etat_qualite = (
                    '<span style="color:red">Alerte Qualité</span>' if quality_alert_info['quality_alert'] == 1 else 'Conforme'
                )

                # Générer la chaîne de résultat formatée
                result_str = (
                    f"**{single_record_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}** - "
                    f"**État Machine :** {etat_machine} (Probabilité : {probability_maintenance:.2f}) | "
                    f"**Qualité Produit :** {etat_qualite}"
                )

                
                st.session_state['stream_results_display'].insert(0, result_str)
                if len(st.session_state['stream_results_display']) > 20:
                    st.session_state['stream_results_display'] = st.session_state['stream_results_display'][:20]

                with placeholder.container():
                    for res in st.session_state['stream_results_display']:
                        st.markdown(res, unsafe_allow_html=True)
                
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"Erreur pendant la simulation de flux : {e}")
            st.session_state['running_stream'] = False
            st.rerun()
            
    if st.session_state.get('stream_results_display'):
        st.subheader("Derniers résultats de la simulation de flux :")
        with placeholder.container():
            for res in st.session_state['stream_results_display']:
                st.markdown(res, unsafe_allow_html=True)

# --- PAGE: Historique des Alertes ---
elif page == "Historique des Alertes":
    st.title("Historique des Alertes Système")
    st.write("Consultez toutes les alertes générées par les modules de maintenance prédictive, de contrôle qualité et de surveillance de la dérive.")

    recent_alerts = get_recent_alerts(num_alerts=200) # Augmenter la limite pour voir plus d'historique
    
    if recent_alerts:
        for alert_line in recent_alerts:
            # Assurer que l'affichage est robuste même si le format du log change légèrement
            display_text = alert_line.strip()
            if "[CRITICAL]" in display_text:
                st.markdown(f"<p style='color:red; font-weight:bold;'>{display_text}</p>", unsafe_allow_html=True)
            elif "[HIGH]" in display_text:
                st.markdown(f"<p style='color:orange;'>{display_text}</p>", unsafe_allow_html=True)
            elif "[WARNING]" in display_text:
                st.markdown(f"<p style='color:goldenrod;'>{display_text}</p>", unsafe_allow_html=True)
            else:
                st.write(display_text)
    else:
        st.info("Aucune alerte n'a été générée pour le moment. Lancez la simulation en direct ou générez des données avec dérive pour voir des alertes.")

# --- PAGE: Configuration des Seuils ---
elif page == "Configuration des Seuils":
    st.title("Configuration Dynamique des Seuils")
    st.write("Ajustez les seuils de détection pour la maintenance prédictive, le contrôle qualité et la surveillance de la dérive. Les modifications sont enregistrées dans la base de données et affecteront les futures analyses.")

    current_config = get_current_config()

    st.subheader("Seuils de Maintenance Prédictive")
    new_ml_prob_threshold = st.slider(
        "Seuil de Probabilité de Défaillance ML (pour alerte 'CRITICAL')",
        min_value=0.0, max_value=1.0, value=float(current_config.get('ml_failure_prob_threshold', 0.5)), step=0.05
    )
    st.info(f"Une alerte 'CRITICAL' de maintenance est déclenchée si la probabilité de défaillance du modèle est supérieure ou égale à {new_ml_prob_threshold:.2f}.")
    
    st.markdown("---")
    st.subheader("Seuils de Contrôle Qualité")
    
    col_ph_min, col_ph_max = st.columns(2)
    with col_ph_min:
        new_ph_min = st.number_input("pH Produit Min.", value=float(current_config.get('ph_product_min_threshold', 6.5)), step=0.1)
    with col_ph_max:
        new_ph_max = st.number_input("pH Produit Max.", value=float(current_config.get('ph_product_max_threshold', 7.5)), step=0.1)
    
    col_color_min, col_color_max = st.columns(2)
    with col_color_min:
        new_color_min = st.number_input("Intensité Couleur Min.", value=float(current_config.get('color_intensity_min_threshold', 100.0)), step=1.0)
    with col_color_max:
        new_color_max = st.number_input("Intensité Couleur Max.", value=float(current_config.get('color_intensity_max_threshold', 200.0)), step=1.0)
        
    col_dim_min, col_dim_max = st.columns(2)
    with col_dim_min:
        new_dim_x_min = st.number_input("Dimension X Min.", value=float(current_config.get('dimension_x_min_threshold', 10.0)), step=0.01)
    with col_dim_max:
        new_dim_x_max = st.number_input("Dimension X Max.", value=float(current_config.get('dimension_x_max_threshold', 10.5)), step=0.01)

    st.markdown("---")
    st.subheader("Seuils pour la Surveillance de Dérive")
    new_drift_detection_percentage = st.slider(
        "Pourcentage d'Écart pour Détecter la Dérive (%)",
        min_value=1.0, max_value=50.0, value=float(current_config.get('drift_detection_percentage', 10.0)), step=1.0
    )
    st.info(f"Une alerte de dérive est déclenchée si la moyenne d'un capteur récent s'écarte de plus de {new_drift_detection_percentage:.2f}% de sa moyenne d'entraînement.")

    st.markdown("---")
    st.subheader("Seuils d'alerte pour la visualisation des machines (non liés au modèle ML)")
    st.info("Ces seuils sont uniquement pour afficher les lignes sur les graphiques dans la page 'Maintenance Prédictive'. Ils ne sont pas utilisés par le modèle ML.")
    new_temp_motor_alert_threshold = st.number_input("Température Moteur (affichage)", value=float(current_config.get('temp_motor_alert_threshold', 90.0)), step=1.0)
    new_vib_motor_alert_threshold = st.number_input("Vibration Moteur (affichage)", value=float(current_config.get('vib_motor_alert_threshold', 3.5)), step=0.1)
    new_pressure_oil_alert_threshold = st.number_input("Pression d'Huile (affichage)", value=float(current_config.get('pressure_oil_alert_threshold', 2.0)), step=0.1)


    if st.button("Enregistrer les Seuils"):
        set_config_value('ml_failure_prob_threshold', new_ml_prob_threshold)
        set_config_value('ph_product_min_threshold', new_ph_min)
        set_config_value('ph_product_max_threshold', new_ph_max)
        set_config_value('color_intensity_min_threshold', new_color_min)
        set_config_value('color_intensity_max_threshold', new_color_max)
        set_config_value('dimension_x_min_threshold', new_dim_x_min)
        set_config_value('dimension_x_max_threshold', new_dim_x_max)
        set_config_value('drift_detection_percentage', new_drift_detection_percentage)
        set_config_value('temp_motor_alert_threshold', new_temp_motor_alert_threshold)
        set_config_value('vib_motor_alert_threshold', new_vib_motor_alert_threshold)
        set_config_value('pressure_oil_alert_threshold', new_pressure_oil_alert_threshold)

        st.cache_data.clear() 
        st.success("Seuils enregistrés avec succès ! Les nouvelles valeurs seront utilisées pour les futures analyses et le contrôle qualité.")
        st.experimental_rerun()
if page == "Jumeau Numérique (Simulation)":
    # Nouveau bloc pour la page Jumeau Numérique
    st.title("⚙️ Jumeau Numérique (Simulation 'What-If')")
    st.write("Expérimentez différents scénarios en ajustant les paramètres de la machine pour voir l'impact sur les prédictions de maintenance et de qualité.")

    st.subheader("Paramètres de la Machine à Simuler")

    # Champs de saisie pour les paramètres
    sim_temp_motor = st.slider("Température Moteur (°C)", min_value=50.0, max_value=120.0, value=85.0, step=0.1)
    sim_vib_motor = st.slider("Vibration Moteur (mm/s)", min_value=1.0, max_value=5.0, value=2.5, step=0.01)
    sim_pressure_oil = st.slider("Pression d'Huile (bar)", min_value=0.5, max_value=4.0, value=1.5, step=0.01)
    sim_num_hours_running = st.slider("Heures de Fonctionnement Cumulées", min_value=0, max_value=10000, value=1000, step=10)

    st.subheader("Paramètres Qualité du Produit")
    sim_ph_product = st.slider("pH Produit", min_value=5.0, max_value=9.0, value=7.0, step=0.01)
    sim_color_intensity = st.slider("Intensité de Couleur", min_value=50.0, max_value=250.0, value=150.0, step=0.1)
    sim_dimension_x = st.slider("Dimension X (mm)", min_value=9.5, max_value=11.0, value=10.2, step=0.001)

    if st.button("Lancer la Simulation"):
        # Importez votre fonction de simulation
        # from src.digital_twin_model import simulate_machine_state
        # Assurez-vous que le fichier src/digital_twin_model.py existe et est dans PYTHONPATH

        try:
            # Appel de la fonction de simulation
            sim_results = simulate_machine_state(
                temp_motor=sim_temp_motor, vib_motor=sim_vib_motor, pressure_oil=sim_pressure_oil,
                ph_product=sim_ph_product, color_intensity=sim_color_intensity, dimension_x=sim_dimension_x,
                num_hours_running=sim_num_hours_running
            )

            st.subheader("Résultats de la Simulation")
            col_maint, col_quality = st.columns(2)

            with col_maint:
                st.metric(
                    label="Probabilité de Défaillance (Maintenance)",
                    value=f"{sim_results['prediction_maintenance_prob']:.2f}"
                )
                if sim_results['prediction_maintenance_prob'] >= get_current_config().get('ml_failure_prob_threshold', 0.5):
                    st.error("🚨 RISQUE ÉLEVÉ DE DÉFAILLANCE PRÉDITE !")
                else:
                    st.success("✅ Faible risque de défaillance")

            with col_quality:
                st.metric(
                    label="Alerte Qualité (Produit)",
                    value="Oui" if sim_results['prediction_quality_alert'] == 1 else "Non"
                )
                if sim_results['prediction_quality_alert'] == 1:
                    st.warning("⚠️ PRODUIT NON-CONFORME PRÉDIT !")
                else:
                    st.info("👍 Qualité conforme prévue")

            st.write("---")
            st.info("Ces résultats sont basés sur les modèles ML actuellement déployés dans MLflow Registry.")

        except Exception as e:
            st.error(f"Erreur lors de la simulation : {e}. Assurez-vous que les modèles MLflow sont chargés et accessibles.")

if page == "Jumeau Numériqu (Simulation)":
    # --- Titre et description avec icône ---
    st.title("⚙️ Jumeau Numérique : Simulateur Prédictif")
    st.markdown("""
    Explorez l'impact des **conditions opérationnelles** et des **paramètres de production** sur la performance future de votre machine et la qualité de votre produit.
    Modifiez les valeurs ci-dessous et lancez la simulation pour obtenir des **prédictions en temps réel** basées sur nos modèles d'apprentissage automatique.
    ---
    """)

    # --- Diagramme d'architecture Mermaid ---
    st.markdown("""
    ```mermaid
    graph TD
        A[Générateur de Données Simulées] --> B(Kafka - topic: sensor_data)
        B --> C[Stream Processor (Python/Pandas/Scikit-learn)]
        C --> D(Base de Données - SQLite/PostgreSQL)
        C --> E(Kafka - topic: maintenance_predictions)
        C --> F(Kafka - topic: quality_alerts)
        E --> D
        F --> D
        D -- Monitoring & Model Training --> G[MLflow Tracking Server]
        G -- Modèles Déployés --> H[MLflow Model Registry]
        D -- Requêtes SQL --> I[Streamlit App - Tableau de Bord]
        D -- Requêtes SQL --> J[Grafana - Tableaux de Bord Avancés & Alertes]
        I -- Simulation ML --> H
    ```
    """)

    # --- Section d'illustration de la machine ---
    st.markdown("""
    <div style="text-align: center;">
        <img src="https://i.ibb.co/3Wf088y/machine-concept-3.png" alt="Concept de machine industrielle futuriste" style="width: 70%; max-width: 600px; border-radius: 10px; box-shadow: 0 0 20px rgba(88, 166, 255, 0.4);">
        <p style="font-size: 0.9em; color: #89b4f9; margin-top: 10px;">Votre réplique numérique interactive pour l'optimisation des opérations.</p>
    </div>
    <br>
    """, unsafe_allow_html=True)

    # --- Conteneur pour les paramètres de simulation ---
    with st.expander("🔬 Configurez votre Scénario de Simulation", expanded=True):
        st.write("Ajustez les curseurs pour définir les conditions de votre scénario.")
        
        col_m_input, col_q_input = st.columns(2)

        with col_m_input:
            st.subheader("Paramètres Opérationnels Machine")
            sim_temp_motor = st.slider("Température Moteur (°C)", min_value=50.0, max_value=120.0, value=85.0, step=0.1, 
                                      help="Température actuelle ou prévue du moteur. Un dépassement peut indiquer un risque de surchauffe.")
            sim_vib_motor = st.slider("Vibration Moteur (mm/s)", min_value=1.0, max_value=5.0, value=2.5, step=0.01, 
                                     help="Niveau de vibration du moteur. Des vibrations excessives sont un signe avant-coureur de défaillance.")
            sim_pressure_oil = st.slider("Pression d'Huile (bar)", min_value=0.5, max_value=4.0, value=1.5, step=0.01, 
                                        help="Pression de l'huile de lubrification. Une pression anormale impacte la durée de vie des composants.")
            sim_num_hours_running = st.slider("Heures de Fonctionnement Cumulées", min_value=0, max_value=10000, value=1000, step=10, 
                                             help="Total des heures de fonctionnement de la machine depuis sa dernière maintenance majeure.")

        with col_q_input:
            st.subheader("Paramètres Qualité Produit")
            sim_ph_product = st.slider("pH Produit", min_value=5.0, max_value=9.0, value=7.0, step=0.01, 
                                      help="Niveau de pH du produit fabriqué. Doit rester dans une plage spécifique pour la conformité.")
            sim_color_intensity = st.slider("Intensité de Couleur", min_value=50.0, max_value=250.0, value=150.0, step=0.1, 
                                           help="Intensité de couleur du produit (valeurs arbitraires). Un écart peut indiquer un défaut esthétique ou fonctionnel.")
            sim_dimension_x = st.slider("Dimension X (mm)", min_value=9.5, max_value=11.0, value=10.2, step=0.001, 
                                       help="Dimension critique du produit. Les tolérances sont souvent très strictes en production.")

    # --- Bouton de simulation ---
    if st.button("🚀 Lancer la Simulation 'What-If'", use_container_width=True):
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner('Analyse du scénario avec les modèles prédictifs...'):
            try:
                # Simulation des résultats (remplacer par votre vrai modèle)
                sim_results = {
                    'prediction_maintenance_prob': min(max(0, (sim_temp_motor - 80) / 40 + (sim_vib_motor - 2) / 3), 1),
                    'prediction_quality_alert': 1 if (sim_ph_product < 6.5 or sim_ph_product > 7.5 or 
                                                    sim_dimension_x < 10.0 or sim_dimension_x > 10.5) else 0
                }

                st.subheader("📊 Résultats de la Simulation Prédictive")
                
                with st.container(border=True):
                    col_res_maint, col_res_quality = st.columns(2)

                    with col_res_maint:
                        st.markdown("### Prédiction Maintenance")
                        ml_prob_threshold = 0.5  # Valeur seuil

                        # Jauge Plotly
                        prob_fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = sim_results['prediction_maintenance_prob'] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Probabilité de Défaillance", 'font': {'size': 18}},
                            delta = {'reference': ml_prob_threshold * 100},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, ml_prob_threshold * 100], 'color': "lightgreen"},
                                    {'range': [ml_prob_threshold * 100, 100], 'color': "red"}],
                                'threshold': {'line': {'color': "black", 'width': 4}, 'value': ml_prob_threshold * 100}
                            }
                        ))
                        st.plotly_chart(prob_fig, use_container_width=True)

                        if sim_results['prediction_maintenance_prob'] >= ml_prob_threshold:
                            st.error("### 🚨 ALERTE : Risque de Défaillance Élevé !")
                            st.write("**Action Recommandée :** Inspection immédiate et planification urgente de la maintenance.")
                        else:
                            st.success("### ✅ STATUT : Faible Risque de Défaillance")
                            st.write("**Statut :** La machine opère dans des conditions optimales.")

                    with col_res_quality:
                        st.markdown("### Prédiction Qualité Produit")
                        
                        if sim_results['prediction_quality_alert'] == 1:
                            st.warning("### ⚠️ ALERTE : Produit Non-Conforme Prédit !")
                            st.write("**Action Recommandée :** Ajustez immédiatement les paramètres de production.")
                        else:
                            st.info("### 👍 STATUT : Qualité Conforme Prévue")
                            st.write("**Statut :** Le produit devrait respecter les spécifications.")

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.subheader("Paramètres Qualité Simulé")
                        st.markdown(f"**pH Produit :** `{sim_ph_product:.2f}` {'🔴' if sim_ph_product < 6.5 or sim_ph_product > 7.5 else '🟢'}")
                        st.markdown(f"**Dimension X :** `{sim_dimension_x:.3f}` {'🔴' if sim_dimension_x < 10.0 or sim_dimension_x > 10.5 else '🟢'}")

                st.markdown("---")
                st.markdown("*Ces résultats sont basés sur une simulation. Pour des prédictions réelles, connectez-vous aux modèles ML en production.*")

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la simulation : {str(e)}")
                st.exception(e)