# Utilise une image Python légère comme base
FROM python:3.9-slim-buster

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers du projet dans le répertoire de travail
COPY . .

RUN mkdir -p /app/data /app/reports

# Exposer le port Streamlit
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["sh", "-c", "python src/bootstrap_model.py && exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]