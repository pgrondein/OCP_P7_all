#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

import gc
from contextlib import contextmanager
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
import itertools
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier

import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import lime
import lime.lime_tabular
import random

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action = 'ignore')


# # Data 

# In[2]:


rs = 42

# Séparation train/test
train_df = pd.read_csv('df_train.csv').drop('Unnamed: 0', axis = 1)
train_df = train_df.set_index('SK_ID_CURR')
valid_df = pd.read_csv('df_valid.csv').drop('Unnamed: 0', axis = 1)
valid_df = valid_df.set_index('SK_ID_CURR')

feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

X_train = train_df[feats]
y_train = train_df['TARGET']

X_valid = valid_df[feats]
y_valid = valid_df['TARGET']


# ## Functions

# In[3]:


#Plot AUC as a function of fold number 
def plot_auc(df, ybot, ytop, model_name):
    
    display(df)
    plt.figure(figsize = (10, 8))
    plt.plot(df, 'bo')
    plt.title('AUC', fontsize = 20)
    plt.ylim(ybot, ytop)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Fold', fontsize = 20)
    plt.ylabel('AUC', fontsize = 20)
    plt.savefig('auc_fold_{}.png'.format(model_name))
    
    plt.show()


# In[4]:


def plot_confusion_matrix(predicted_labels_list, y_test_list, model_name):
    
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision = 2)
    class_names = train_df['TARGET'].unique()
 
    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize = (8,8))
    generate_confusion_matrix(cnf_matrix, 
                              classes = class_names, 
                              title = 'Confusion matrix')
    plt.show()
        
def generate_confusion_matrix(cnf_matrix, classes = feats, normalize = False, title = 'Confusion matrix'):
    
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation = 'nearest', cmap = plt.get_cmap('Blues'))
    plt.title(title, fontsize = 20)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize = 15)
    plt.yticks(tick_marks, classes, fontsize = 15)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment = "center",
                 color = "white" if cnf_matrix[i, j] > thresh else "black", fontsize = 15)

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 20)
    plt.xlabel('Predicted label', fontsize = 20)

    return cnf_matrix


# In[5]:


#F1_score as a function of threshold
def f1_max(y_valid, y_pred, y_score, model_name):
    
    list_threshold = np.arange(0, 1, 0.01).tolist()
    list_f1_score = []
    
    for threshold in list_threshold:
        y_pred = y_score > threshold
        list_f1_score.append(f1_score(y_valid, y_pred))
        
    y_max = max(list_f1_score)
    x_pos = list_f1_score.index(y_max)
    x_max = list_threshold[x_pos]
    
    plt.figure(figsize = (10, 8))
    plt.plot(list_threshold, list_f1_score)
    plt.title('F1 threshold', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Threshold', fontsize = 20)
    plt.xlim(0, 1)
    plt.ylim(0, y_max + 0.1)
    plt.ylabel('F1 score', fontsize = 20)
    plt.vlines(x = x_max, ymin = 0, ymax = y_max, color = 'r')
    plt.hlines(y = y_max, xmin = 0, xmax = x_max, color = 'r')
    plt.savefig('f1_threshold_{}.png'.format(model_name))
    plt.show()
    
    print('F1 maximum', round(y_max, 2),' atteint pour threshold = ', x_max)
    
    return x_max, y_max


# In[6]:


def pie(df, var, lim):
    
    df_plot = pd.DataFrame(df[var].value_counts(normalize = True))
    df_plot = df_plot.rename(columns = {var :'Frequence' })
    df_plot_1 = df_plot.loc[df_plot['Frequence'] > lim]
    
    plot = df_plot_1.plot(kind = 'pie', y = 'Frequence', autopct = '%1.0f%%', figsize = (10, 10), 
                          fontsize = 25, legend = False, labeldistance = 1.1)
                                                     
    plot.set_title(var, fontsize = 40)
    plt.title(var,fontsize = 30)
    plt.xlabel('')
    plt.ylabel('')
    plt.axis('equal') 
    plt.show()
    
    pct = pd.DataFrame(data = df[var].value_counts()).rename(columns = {var :'Nb Users' })
    pct['% users'] = round((pct['Nb Users']/len(df))*100, 2)
    
    display(pct)


# In[54]:


# Display feature importance
def feat_importance(model_name, model):
    
    if model_name == 'LR_clf' :
        coef = model.coef_
        feat_imp = pd.DataFrame(data = {'Features' : feats, 'Importance' : coef[0]})
        
        feat_imp.sort_values('Importance', key = abs, ascending = False, inplace = True)
        plt.figure(figsize = (10,10))
        sns.barplot(y = feat_imp['Features'].head(10), 
                    x = feat_imp['Importance'].head(10), 
                    data = feat_imp)
    
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.xlabel('Importance',fontsize = 20)
        plt.ylabel('',fontsize = 20)
    
        plt.title('Feature Importance {}'.format(model_name), fontsize = 20)
        plt.savefig('feat_imp_{}.png'.format(model_name))
        plt.show()
        
    if model_name == 'RF_clf' : 
        
        coef = model.feature_importances_
#         display(coef)
        feat_imp = pd.DataFrame(data = {'features' : feats, 'Importances' : coef})
        
        cols = feat_imp[
               ["features", "Importances"]].groupby("features").mean().sort_values(
               by = "Importances", ascending = False)[:40].index
    
        best_features = feat_imp.loc[feat_imp.features.isin(cols)]
    
        plt.figure(figsize = (8, 10))
        sns.barplot(x = "Importances", y = "features", data = best_features.sort_values(by = "Importances", 
                                                                                      ascending = False))
        plt.title('Features Importance {}'.format(model_name), fontsize = 20)
        plt.savefig('feat_imp_{}.png'.format(model_name))


# In[8]:


# Grid Search of model on train dataset function
def model_GS(X_train, y_train, model, model_name, grid, class_imb = 'CW', ybot = 0, ytop = 1, 
             n_fold = 5, n_fold_repeats = 1) : 
    
   # IMBALANCED CLASS METHOD
    if class_imb == 'SMOTE' :       
        oversample_model = SMOTE()   
        steps = [
        ('oversample', oversample_model), 
        ('model', model)
        ]
        
    if class_imb == 'CW' :  
        steps = [
            ('model', model)
        ]
    pipeline = Pipeline(steps = steps)
    
    
    # CROSS VALIDATION METHOD : RepeatedStratifiedKFold + GridSearch
    cv = RepeatedStratifiedKFold(n_splits = n_fold, 
                                 n_repeats = n_fold_repeats, 
                                 random_state = rs) 
    model_cv = GridSearchCV(pipeline, 
                            grid, 
                            cv = cv,
                            scoring = 'roc_auc',
                            refit = True)
    best_model = model_cv.fit(X_train, y_train)
        
    cv_result = pd.DataFrame(data = model_cv.cv_results_)
    cv_result.to_csv('cv_result_{}.csv'.format(model_name))

    best_params = pd.DataFrame(data = model_cv.best_params_, index = [model_name])
    best_params.to_csv('best_params_{}.csv'.format(model_name))
        
    b = n_fold * n_fold_repeats
    best_i = model_cv.best_index_
    auc = pd.DataFrame(columns = ['Fold', 'AUC'])
        
    for i in range (0, b) :
        best_auc = model_cv.cv_results_['split{}_test_score'.format(i)][best_i]
        auc = auc.append({'Fold' : i+1, 'AUC' : best_auc}, ignore_index = True)
    
    print('Best parameters : ', model_cv.best_params_)
    print('Mean computation time : %.3f secondes' % model_cv.refit_time_)
    print('Mean AUC: %.3f' % model_cv.cv_results_['mean_test_score'][best_i])
    print('Standard deviation AUC: %.3f' % model_cv.cv_results_['std_test_score'][best_i])
              
    auc = auc.set_index('Fold')
    plot_auc(auc, ybot, ytop, model_name)
     
    return best_model


# In[9]:


# Model validation on valid dataset
def model_validation(model, model_name, X_valid, y_valid):
    
    y_pred = model.predict(X_valid)
    y_score = model.predict_proba(X_valid)[:,1]
    
    auc = roc_auc_score(y_valid, y_score)
    f1 = f1_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    prec = precision_score(y_valid, y_pred)
    
    scores = pd.DataFrame({'AUC' : round(auc, 2), 
                            'F1' : round(f1, 2), 
                            'Recall score' : round(recall, 2), 
                            'Precision' : round(prec, 2)}, 
                           index = [model_name])
    display(scores)
    scores.to_csv('score_{}.csv'.format(model_name))
    
    plot_confusion_matrix(y_pred, y_valid, model_name)
    
    if (model_name == 'LR_clf') or (model_name == 'RF_clf'):
        
        hist_fig = plt.figure(figsize = (10, 8))
        plt.hist(y_score, bins = 'auto')
        plt.title('Proba distribution', fontsize = 20)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.xlabel('y_score', fontsize = 20)
        plt.savefig('hist_proba_{}.png'.format(model_name))
        plt.show()
        
        thres, max_F1 = f1_max(y_valid, y_pred, y_score, model_name)
        y_proba = model.predict_proba(X_valid)
        y_pred = [1 if y.max() > thres else 0 for y in y_proba]
        y_pred_df = pd.DataFrame(data = y_pred, columns = ['TARGET'])
        pie(y_pred_df, 'TARGET', 0)
        
        return scores, thres, max_F1
    
    else : 
        return scores


# # Models

# ## Dummy Classifier

# In[57]:


# Model definition
dum = DummyClassifier(strategy = "most_frequent")
dum.fit(X_train, y_train)
score_dum = model_validation(dum, 'Dummy_clf', X_valid, y_valid)


# ## Logistic Regression

# In[305]:


LR = LogisticRegression(class_weight = 'balanced')

grid_LR = {"model__C" : [0.1, 1, 10, 100], 
           "model__penalty" : ["l1","l2"]}

LR_cv = model_GS(X_train, y_train, LR, 'LR_clf', grid = grid_LR, class_imb = 'CW')


# In[61]:


best_param_LR = pd.read_csv('best_params_LR_clf.csv')
best_C = float(best_param_LR['model__C'])
best_pen = best_param_LR['model__penalty'][0]

LR_cv = LogisticRegression(class_weight = 'balanced', C = best_C, penalty = best_pen)
LR_cv.fit(X_train, y_train)

feat_importance('LR_clf', LR_cv)

score_LR, thres_LR, max_F1_LR = model_validation(LR_cv, 'LR_clf', X_valid, y_valid)


# In[15]:


LR_explainer = lime.lime_tabular.LimeTabularExplainer(X_valid.values,
                                                      mode = 'classification', 
                                                      feature_names = feats,
                                                      class_names  = list(y_valid.unique()),
                                                      random_state = rs
                                                      )


# In[16]:


idx = random.randint(1, len(X_valid))
print('Demande de prêt : ', idx)

print("Prediction : ", 
      "Accepté" if LR_cv.predict_proba(X_valid.values[idx].reshape(1,-1))[0].max() > thres_LR else 'Rejeté')
print("Actual :     ", "Accepté" if y_valid.iloc[idx] == 0 else 'Rejeté')

LR_explanation = LR_explainer.explain_instance(X_valid.values[idx],
                                               LR_cv.predict_proba,
                                               num_features = 10,
                                               )
LR_explanation.show_in_notebook()


# In[17]:


plt.rcParams["figure.figsize"] = [10,8]
with plt.style.context("ggplot"):
    LR_explanation.as_pyplot_figure()


# ## Random Forest Classifier 

# ### OOB error graph

# In[326]:


# Hyper parameters tuning (tree number n)
min_estimators = 10
max_estimators = 200
n_estimators = [i for i in range(min_estimators, max_estimators+1, 10)]

# Define a pipeline that first transform training with SMOTE set then fits the model
forest = RandomForestClassifier(warm_start = False, 
                                oob_score = True,
                                max_features = 'sqrt', 
                                random_state = rs,
                                class_weight = 'balanced')

# OOb error as a function of tree number n
oob_error = pd.DataFrame(columns = ['n','oob'])    
for i in n_estimators:
    print('estimator :', i)
    forest.set_params(n_estimators = i)
    forest.fit(X_train, y_train)
    d = {'n' : i,'oob' : 1 - forest.oob_score_}
    oob_error = oob_error.append(d, ignore_index = True)


# In[327]:


plt.figure(figsize = (12,10))
ax = plt.axes()
ax.set_ylim(0, 0.5)
            
plt.plot(oob_error['n'], oob_error['oob'])
plt.title(('oob error '), fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("n_estimators", fontsize = 20)
plt.ylabel("OOB error rate", fontsize = 20)
plt.show()

plt.savefig('oob_error.png')
plt.show()


# ### Model

# In[19]:


n_var = X_train.shape[1]
forest = RandomForestClassifier(warm_start = False, 
                                oob_score = True,
                                max_features = 'sqrt', 
                                random_state = rs,
                                class_weight = 'balanced')
grid_RF = {
    'model__n_estimators' : [50, 70, 90],
    'model__min_samples_leaf' : [2, 5, 10],
    'model__max_depth': [int(n_var/2), int(n_var/3), int(n_var/4)]
}

RF_cv = model_GS(X_train, y_train, forest, 'RF_clf', grid = grid_RF, class_imb = 'CW')


# In[63]:


# best_param_RF = pd.read_csv('best_params_RF_clf.csv')
# best_n = int(best_param_RF['model__n_estimators'])
# best_msl = float(best_param_RF['model__min_samples_leaf'])
# best_md = int(best_param_RF['model__max_depth'])

# RF_best = RandomForestClassifier(max_depth = best_md,
#                                  min_samples_leaf = best_msl,
#                                  n_estimators = best_n,
#                                  warm_start = False, 
#                                  oob_score = True,
#                                  max_features = 'sqrt', 
#                                  random_state = rs,
#                                  class_weight = 'balanced')
# RF_best.fit(X_train, y_train)

feat_importance('RF_clf', RF_cv.best_estimator_.named_steps['model'])

score_RF, thres_RF, max_F1_RF = model_validation(RF_cv.best_estimator_.named_steps['model'], 
                                                'RF_clf', X_valid, y_valid)


# # Conclusion

# In[64]:


clf_results = pd.concat([score_dum, score_LR, score_RF], axis = 0)
clf_results.to_csv('results_clf.csv')

# clf_results = pd.read_csv('results_clf.csv', index_col = 'Unnamed: 0')
clf_results.style.highlight_max(color = 'red', axis = 0).set_precision(2)


# In[ ]:




