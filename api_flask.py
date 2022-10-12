# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:17:07 2022

@author: Pask
"""
import pandas as pd
import dill as pickle
import json
from flask import Flask, jsonify, request
from functions import predict

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def get_prediction():
    idx = request.get_json()
    rep, proba = predict(idx, df_test, best_model, thres, scaler)
    d = {
        'rep' : rep,
        'proba' : proba
        }
 
    return jsonify(d)
 
if __name__ == '__main__':
    #chargement du scaler entraîné
    scaler = pickle.load(open('MinMaxScaler_LR.sav', 'rb'))
    #chargement des données Test
    df_test = pd.read_csv('df_test.csv').drop('Unnamed: 0', axis = 1)
    df_test = df_test.set_index('SK_ID_CURR')
    #chargement du meilleur modèle
    best_model = pickle.load(open('LR_clf.sav', 'rb'))
    #chargement du palier de probabilité
    f = open('LR_params.json')
    thres_dict = json.load(f)
    thres = thres_dict['Threshold']
    
    
    app.run(debug=True, host='0.0.0.0')