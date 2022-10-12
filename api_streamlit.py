# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(layout="wide")
import dill as pickle
import json
import pandas as pd
import plotly.graph_objects as go
import requests

from functions import feat_loc, feat_glo, local_css, dist_proba
local_css("style.css")

url = 'http://192.168.1.37:5000/predict'

#chargement des données Valid/Test
df_valid = pd.read_csv('df_valid.csv').drop('Unnamed: 0', axis = 1)
df_valid = df_valid.set_index('SK_ID_CURR')
feats = [f for f in df_valid.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
X_valid = df_valid[feats]
y_valid = df_valid['TARGET']
df_test = pd.read_csv('df_test.csv').drop('Unnamed: 0', axis = 1)
df_test = df_test.set_index('SK_ID_CURR')
#chargement du meilleur modèle
best_model = pickle.load(open('LR_clf.sav', 'rb'))
#chargement des coefficients globaux
f = open('LR_coef_global.json')
coef = json.load(f)
coef_global = pd.DataFrame(data = coef)
#chargement du explainer Lime entraîné
explainer = pickle.load(open('LR_explainer.sav', 'rb'))
#chargement du palier de probabilité
f = open('LR_params.json')
thres_dict = json.load(f)
thres = thres_dict['Threshold']
#Chargement df description variables
df_desc = pd.read_csv('HomeCredit_columns_description.csv', encoding= 'unicode_escape')



#Titre de la page
st.title("Projet 7 - Implémentez un modèle de scoring")

#Menu déroulant
values = df_test.index.tolist()
values.insert(0, '<Select>')
num = st.sidebar.selectbox(
    "Veuillez sélectionner un numéro de demande de prêt",
    values
)
if num != '<Select>':
    idx = num
    j_idx = json.dumps(idx)
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    request = requests.post(url, data = j_idx, headers = headers)
    req = request.json()
    proba_api = req['proba']
    rep_api = req['rep']

    #Prédiction
    if rep_api == 'Acceptée':
        t = "<span class='highlight blue'><span class='bold'>Acceptée</span></span>"
    else : 
        t = "<span class='highlight red'><span class='bold'>Rejetée</span></span>"
    
    #Présentation des résultats
    st.markdown(
        """
        <style>
        .header-style {
            font-size:25px;
            font-family:sans-serif;
        }
        </style>
        """,
        unsafe_allow_html = True
    )
    st.markdown(
        """
        <style>
        .font-style {
            font-size:20px;
            font-family:sans-serif;
        }
        </style>
        """,
        unsafe_allow_html = True
    )
    st.markdown(
        '<h2> Statut de la demande </h2>',
        unsafe_allow_html = True
    )
    column_1, column_2 = st.columns(2)
    column_1.markdown(
        '<h3>Décision</h3>',
        unsafe_allow_html = True
    )
    column_1.markdown(t, unsafe_allow_html = True)

    column_2.markdown(
        '<h3>Score</h3>',
        unsafe_allow_html = True
    )
    column_2.subheader(f"{proba_api}")
    #Jauge 
    gauge = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = proba_api,
        mode = "gauge+number",
        title = {'text': "Score", 'font': {'size': 24}},
        gauge = {'axis': {'range': [None, 1]},
                 'bar': {'color': "grey"},
                 'steps' : [
                     {'range': [0, 0.67], 'color': "lightblue"},
                     {'range': [0.67, 1], 'color': "lightcoral"}],
                 'threshold' :
                     {'line': {'color': "red", 'width': 4}, 
                      'thickness': 1, 'value': thres,
                     }
                }
            ))

    st.plotly_chart(gauge)
    
    
  #Importance des variables
    st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html = True
    )
    st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html = True
    )
    nb_ = st.slider('Variables à visualiser', 0, 25, 10)
    if 'key' not in st.session_state:
        st.session_state['nb_var'] = nb_
    else : 
        st.session_state['nb_var'] = nb_
            
    nb = st.session_state['nb_var']

    column_1, column_2 = st.columns(2)
    #Détails individuels
    column_1.markdown(
            '<h3>Détails individuels</h3>',
            unsafe_allow_html = True
    )
    coef_loc = feat_loc(idx, df_test, best_model, explainer, nb)
    column_1.plotly_chart(coef_loc, use_container_width = True)
    #Détails globaux
    column_2.markdown(
            '<h3>Détails globaux</h3>',
            unsafe_allow_html = True
    )
    coef_glo = feat_glo(coef_global, nb)
    column_2.plotly_chart(coef_glo, use_container_width = True)
        
    #Choix de la variables à expliquée
    feats = df_desc['Row'].tolist()
    var = st.selectbox(
            "Veuillez sélectionner une variable à définir",
            feats
    )
    d = df_desc.loc[df_desc['Row'] == var]
    st.write('Variable {} : {}'.format(var, d['Description'].values[0]))
    
    #Bouton pour voir distribution
    st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html = True
    )
    st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html = True
    )

    if st.button('Voir distribution par classes'):
        st.markdown(
        '<h2> Distribution des probabilités </h2>',
        unsafe_allow_html = True
        )
        hist = dist_proba(X_valid, best_model, proba_api)
        st.plotly_chart(hist)
        