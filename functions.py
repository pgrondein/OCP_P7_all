# -*- coding: utf-8 -*-

#Fonctions pour l'api P7

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go

def as_pyplot_figure(explanation, label = 1, figsize = (10,10)):
    
    exp = explanation.as_list(label = label)
    fig = plt.figure(figsize = figsize)
    coefs = [x[1] for x in exp]
    feats = [x[0] for x in exp]
    coefs.reverse()
    feats.reverse()
    feats = [f.strip('<= 0.00 0.46') for f in feats]
    feats = [f[:25] for f in feats]
    
    colors = ['Défavorable' if x > 0 else 'Favorable' for x in coefs]

    fig = px.bar( 
                 x = coefs, 
                 y = feats,
                 color = colors,
                 color_discrete_map = {
                     'Favorable': 'lightblue',
                     'Défavorable': 'lightcoral'
                 },
                 width = 800, 
                 height = 800
                 )
    fig.update_layout(showlegend = False)
    fig.update_yaxes(title='Variables', 
                     visible = True, 
                     showticklabels = True)
    fig.update_xaxes(title='Importance', 
                     visible = True, 
                     showticklabels = True)
    fig.update_layout(yaxis = {'categoryorder':'array', 'categoryarray': feats})
    
    return fig

def feat_loc(idx, df_test, best_model, explainer, nb) :
    #Calcul des coefs locaux pour les variables
    explanation = explainer.explain_instance(np.array(df_test.loc[idx]),
                                             best_model.predict_proba,
                                             num_features = nb
                                             )
    loc_coef = as_pyplot_figure(explanation)
    
    return loc_coef

def feat_glo(df, nb):
    df.sort_values('Importance', 
                   key = abs, 
                   ascending = False, 
                   inplace = True)
    feats = df['Features'].head(nb).to_list()
    feats = [f.strip('<= 0.00 0.46') for f in feats]
    feats = [f[:25] for f in feats]
    coefs = df['Importance'].head(nb).to_list()
    coefs.reverse()
    feats.reverse()
#    st.dataframe(df.head(nb))
    
    colors = ['Défavorable' if x > 0 else 'Favorable' for x in coefs]
    
    fig = px.bar( 
                 x = coefs, 
                 y = feats,
                 color = colors,
                 color_discrete_map = {
                     'Favorable': 'lightblue',
                     'Défavorable': 'lightcoral'
                 },
                 width = 800, 
                 height = 800
                 )
    fig.update_layout(showlegend = False)
    fig.update_layout(yaxis = {'categoryorder':'array', 
                               'categoryarray': feats})
    fig.update_yaxes(title='Variables', 
                     visible = True, 
                     showticklabels = True)
    fig.update_xaxes(title='Importance', 
                     visible = True, 
                     showticklabels = True)
    return fig

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def predict(idx, df_test, best_model, thres, scaler) :
    #Récupération de la demande de prêt
    ind = np.array(df_test.loc[idx]).reshape(1,-1)
    #Calcul de la probabilité d'un rejet
    pipe = Pipeline([('scaler', scaler), ('model', best_model)])
    proba = float(pipe.predict_proba(ind)[:,1])
    
    rep = "Rejetée" if proba > thres else 'Acceptée'
    
    return rep, round(proba, 2)

def dist_proba(X, best_model, proba_api):
    y_proba_0 = best_model.predict_proba(X)[:,0]
    y_proba_1 = best_model.predict_proba(X)[:,1]
    
    fig = go.Figure(layout_yaxis_range=[0,1200])
    fig.add_trace(go.Histogram(x = y_proba_0,
                               name = 'Acceptée',
                               marker_color = 'lightblue'
                               ))
    fig.add_trace(go.Histogram(x = y_proba_1,
                               name = 'Rejetée',
                               marker_color = 'lightcoral'
                               ))
    fig.add_trace(
        go.Scatter(
            x = [proba_api, proba_api],
            y = [0, 1200],
            mode = "lines",
            line = go.scatter.Line(color = "black"),
            showlegend = True,
            name = 'Individu'
            )
        )
    fig.update_layout(barmode ='overlay',
                      xaxis_title_text = 'Probabilité',
                      yaxis_title_text = 'Count',
                      width = 1200, 
                      height = 800,
                      xaxis = {'tickfont_size': 15},
                      yaxis = {'tickfont_size': 15},
                      font = dict(
                          size = 15
                          )
                     )
    fig.update_traces(opacity = 0.75)
    
    return fig
    