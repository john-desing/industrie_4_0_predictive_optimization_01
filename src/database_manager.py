import sqlite3
import pandas as pd
import datetime
import os

DATABASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'industry_data.db')

def get_db_connection():
    """Établit une connexion à la base de données SQLite."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row # Permet d'accéder aux colonnes par leur nom
    return conn

def create_tables():
    """Crée les tables nécessaires dans la base de données si elles n'existent pas."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table pour les lectures de capteurs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            temperature_motor REAL,
            vibration_motor REAL,
            pressure_oil REAL,
            ph_product REAL,
            color_intensity REAL,
            dimension_x REAL
        );
    """)

    # Table pour les prédictions de maintenance prédictive
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            prediction INTEGER, -- 0: normal, 1: failure risk
            probability REAL,
            FOREIGN KEY (reading_id) REFERENCES sensor_readings(id)
        );
    """)

    # Table pour les alertes de contrôle qualité
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quality_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            quality_alert INTEGER, -- 0: conforme, 1: alerte
            ph_out_of_spec INTEGER,
            color_out_of_spec INTEGER,
            dimension_x_out_of_spec INTEGER,
            FOREIGN KEY (reading_id) REFERENCES sensor_readings(id)
        );
    """)

    # NOUVEAU: Table pour les configurations de seuils
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value REAL
        );
    """)
    
    conn.commit()
    conn.close()
    
    # Initialiser les seuils par défaut si la table est vide
    initialize_default_thresholds()

def initialize_default_thresholds():
    """Insère les seuils par défaut si la table system_config est vide."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    defaults = {
        'temp_motor_alert_threshold': 90.0,
        'vib_motor_alert_threshold': 3.5,
        'pressure_oil_alert_threshold': 2.0,
        'ph_product_min_threshold': 6.5,
        'ph_product_max_threshold': 7.5,
        'color_intensity_min_threshold': 100.0,
        'color_intensity_max_threshold': 200.0,
        'dimension_x_min_threshold': 10.0,
        'dimension_x_max_threshold': 10.5,
        'ml_failure_prob_threshold': 0.5, # Seuil de probabilité pour la prédiction ML
        'drift_detection_percentage': 10.0 # Seuil de dérive en %
    }
    
    for key, default_value in defaults.items():
        cursor.execute("INSERT OR IGNORE INTO system_config (key, value) VALUES (?, ?)", (key, default_value))
    
    conn.commit()
    conn.close()

def get_config_value(key):
    """Récupère une valeur de configuration par sa clé."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM system_config WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result['value'] if result else None

def set_config_value(key, value):
    """Met à jour une valeur de configuration."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO system_config (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def get_all_config_values():
    """Récupère toutes les valeurs de configuration."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM system_config")
    config = {row['key']: row['value'] for row in cursor.fetchall()}
    conn.close()
    return config

# Les fonctions existantes restent inchangées:
# insert_sensor_data, fetch_sensor_data, insert_maintenance_prediction, fetch_maintenance_predictions,
# insert_quality_alert, fetch_quality_alerts

def insert_sensor_data(df):
    """Insère les données de capteurs d'un DataFrame dans la table sensor_readings."""
    conn = get_db_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO sensor_readings (timestamp, temperature_motor, vibration_motor, pressure_oil, ph_product, color_intensity, dimension_x)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            row['temperature_motor'],
            row['vibration_motor'],
            row['pressure_oil'],
            row['ph_product'],
            row['color_intensity'],
            row['dimension_x']
        ))
    conn.commit()
    conn.close()

def fetch_sensor_data(limit=None):
    """Récupère les données brutes des capteurs de la base de données."""
    conn = get_db_connection()
    query = "SELECT * FROM sensor_readings ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def insert_maintenance_prediction(reading_id, timestamp, prediction, probability):
    """Insère une prédiction de maintenance prédictive."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO maintenance_predictions (reading_id, timestamp, prediction, probability)
        VALUES (?, ?, ?, ?)
    """, (reading_id, timestamp.strftime('%Y-%m-%d %H:%M:%S'), int(prediction), float(probability)))
    conn.commit()
    conn.close()

def fetch_maintenance_predictions(limit=None):
    """Récupère les prédictions de maintenance prédictive."""
    conn = get_db_connection()
    query = "SELECT * FROM maintenance_predictions ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def insert_quality_alert(reading_id, timestamp, quality_alert, ph_out_of_spec, color_out_of_spec, dimension_x_out_of_spec):
    """Insère une alerte de contrôle qualité."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO quality_alerts (reading_id, timestamp, quality_alert, ph_out_of_spec, color_out_of_spec, dimension_x_out_of_spec)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (reading_id, timestamp.strftime('%Y-%m-%d %H:%M:%S'), quality_alert, ph_out_of_spec, color_out_of_spec, dimension_x_out_of_spec))
    conn.commit()
    conn.close()

def fetch_quality_alerts(limit=None):
    """Récupère les alertes de contrôle qualité."""
    conn = get_db_connection()
    query = "SELECT * FROM quality_alerts ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

if __name__ == "__main__":
    # Test des nouvelles fonctions
    create_tables()
    print("Tables créées et seuils par défaut initialisés.")
    
    # Test de lecture
    all_config = get_all_config_values()
    print("\nConfiguration actuelle:")
    for k, v in all_config.items():
        print(f"  {k}: {v}")
    
    # Test de modification
    set_config_value('temp_motor_alert_threshold', 95.0)
    print(f"\nSeuil de température moteur mis à jour à: {get_config_value('temp_motor_alert_threshold')}")
    
    # Réinitialisation si vous voulez
    # if os.path.exists(DATABASE_PATH):
    #     os.remove(DATABASE_PATH)
    # create_tables()
    # print("DB réinitialisée et recréée avec seuils par défaut.")