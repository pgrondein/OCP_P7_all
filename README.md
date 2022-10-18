# OCP_P7_all
# P7 - Implémentez un modèle de scoring - All files 
## Déploiement d'un dashboard interrogeant un modèle de scoring (OpenClassrooms | Data Scientist | Projet 7)

L’entreprise souhaite développer un modèle de scoring de la probabilité de défaut de paiement du client pour étayer la décision d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

Ce projet est découpé en trois parties.

### 1. Partie préparation des données et modélisation
- dossier_code_prepa.py : préparation des données 
- dossier_code_model.py : test de différents modèle de ML et choix du meilleur pour notre cas
    
### 2. Partie API Flask (api_flask.py)
- Hébergée par Heroku 
- Dépôt Github disponible ici https://github.com/pgrondein/OCP_P7_heroku
- Api visible ici https://apip7heroku.herokuapp.com/
- Elle prend en entrée le numéro de demande de prêt et renvoie la probabilité pour la classe 1 (rejetée) ainsi que le statut de la demande.

### 3. Partie Dashboard Streamlit (api_streamlit.py)
- Hébergé par Streamlit share 
- Dépôt Github disponible ici https://github.com/pgrondein/OCP_P7_streamlitshare
- Dashboard visible ici https://pgrondein-ocp-p7-streamlitshare-api-streamlit-mbxmbt.streamlitapp.com/
- Affiche différents éléments :
    - Choix d'un numéro de demande dans menu déroulant
    - Décision du modèle et score, obtenus en appelant l'API Flask
    - Positionnement par rapport au seuil lié à la fonction coût métier
    - Importance globale et locale des paramètres dans la décision du modèle
    - Positionnement de la demande par rapport à la distribution de la probabilité pour la classe 1 (rejetée)
                                                            
