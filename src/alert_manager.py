import datetime
import os

ALERT_LOG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'reports', 'alerts.log')

def log_alert(alert_type, message, severity="INFO"):
    """
    Simule l'envoi d'une alerte en l'écrivant dans un fichier log.
    Dans une application réelle, cela enverrait un email, un SMS, etc.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [{severity}] [{alert_type}] {message}\n"
    
    # Assurez-vous que le répertoire 'reports' existe
    os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)
    
    with open(ALERT_LOG_FILE, 'a') as f:
        f.write(log_entry)
    
    print(f"ALERTE LOGGÉE: {alert_type} - {message}")

def get_recent_alerts(num_alerts=20):
    """
    Récupère les dernières alertes du fichier log.
    """
    if not os.path.exists(ALERT_LOG_FILE):
        return []
    
    with open(ALERT_LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    return lines[-num_alerts:][::-1] # Retourne les N dernières lignes dans l'ordre inverse (plus récent en premier)

if __name__ == "__main__":
    # Test
    log_alert("Maintenance", "Haute vibration détectée sur le moteur M-001.", "HIGH")
    log_alert("Qualité", "pH produit hors spécifications pour le lot L-ABC.", "MEDIUM")
    print("\nDernières alertes:")
    for alert in get_recent_alerts():
        print(alert.strip())