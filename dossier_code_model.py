#!/usr/bin/env python
# coding: utf-8

# In[16]:


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

import lime
import shap

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action = 'ignore')


# # Data 

# In[2]:


rs = 42
df = pd.read_csv('data_clean.csv')

# Séparation train/test
train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]

feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

X_train = train_df[feats]
y_train = train_df['TARGET']

scaler = MinMaxScaler()
X_train = pd.DataFrame(data = scaler.fit_transform(X_train), 
                       columns = X_train.columns, index = X_train.index)

print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))


# In[3]:


np.all(np.isfinite(train_df[feats]))


# In[4]:


np.all(np.isnan(train_df[feats]))


# ## Functions

# In[5]:


def display_importances(feature_importance_df_, model):
    
    cols = feature_importance_df_[
        ["feature", "importance"]].groupby("feature").mean().sort_values(
        by = "importance", ascending = False)[:40].index
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize = (8, 10))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", 
                                                                                  ascending = False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('{}_importances.png'.format(model))


# In[6]:


def smote_pip_skfold(X_train, y_train, model, model_name, rs = 42) : 
    
    temps1 = time.time()
    
    # Define SMOTE version of the dataset
    oversample_model = SMOTE(sampling_strategy = 'auto', 
                             k_neighbors = 5)
    
    # Define a pipeline that first transform training set with SMOTE set then fits the model
    steps = [('oversample', oversample_model), 
             ('model', model)]
    pipeline = Pipeline(steps = steps)
    
    # Define cross validation method
    cv = RepeatedStratifiedKFold(n_splits = 5, 
                                 n_repeats = 3, 
                                 random_state = rs)
    
    scores = cross_validate(pipeline, 
                            X_train, y_train, 
                            cv = cv,
                            scoring = ('roc_auc', 'f1'),
                            return_train_score = False)
    
    print('Mean AUC: %.3f' % np.mean(scores['test_roc_auc']))
    print('Mean F1 score: %.3f' % np.mean(scores['test_f1']))
    
    duration1 = time.time() - temps1
    print("Computation time : ", "%15.2f" % duration1, "secondes")
    
    data_clf = pd.DataFrame(data = {'AUC' : np.mean(scores['test_roc_auc']), 
                                    'F1' : np.mean(scores['test_f1']), 
                                    'Computation time' : round(duration1, 0)}, 
                             index = [model_name])
    
    
    display(data_clf)
    
    return data_clf


# In[33]:


def evaluate_model(model) : 
    
    folds = StratifiedKFold(n_splits = 5, 
                            shuffle = True, 
                            random_state = 1)
    
    predicted_targets = np.array([])
    actual_targets = np.array([])
    
    train_df = df[df['TARGET'].notnull()]   
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    oversample_model = SMOTE(sampling_strategy = 'auto', 
                             k_neighbors = 5)
    
    for n_fold, (train_idx, test_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        test_x, test_y = train_df[feats].iloc[test_idx], train_df['TARGET'].iloc[test_idx]
        
        train_x, train_y = oversample_model.fit_resample(train_x, train_y)
        
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        
        predicted_targets = np.append(predicted_targets, y_pred)
        actual_targets = np.append(actual_targets, test_y)
        
    return predicted_targets, actual_targets

def plot_confusion_matrix(predicted_labels_list, y_test_list):
    
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision = 2)
    
    class_names = train_df['TARGET'].unique()
 
    # Plot non-normalized confusion matrix
    plt.figure(figsize = (8,8))
    generate_confusion_matrix(cnf_matrix, 
                              classes = class_names, 
                              title = 'Confusion matrix, without normalization')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure(figsize = (8,8))
    generate_confusion_matrix(cnf_matrix, 
                              classes = class_names, 
                              normalize = True, 
                              title = 'Normalized confusion matrix')
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


# # Models

# ## Dummy Classifier

# In[85]:


# Model definition
dum = DummyClassifier(strategy = "most_frequent")

dummy_clf = smote_pip_skfold(X_train, y_train, dum, 'Dummy_clf')


# ## Logistic Regression with Grid Search & SMOTE

# ### Hyper parameters tuning

# In[10]:


# Hyper parameters tuning with Grid Search & SMOTE method

# Define SMOTE version of the dataset
oversample_model = SMOTE(sampling_strategy = 'auto', 
                         k_neighbors = 5)

# Define a pipeline that first transform training with SMOTE set then fits the model
steps = [('oversample', oversample_model), 
         ('model', LogisticRegression())]
pipeline = Pipeline(steps = steps)

# Define cross validation method
cv = RepeatedStratifiedKFold(n_splits = 5, 
                             n_repeats = 3, 
                             random_state = rs)

# Define Grid search method
grid = {"model__C" : [0.01, 0.1, 1, 10, 100], 
        "model__penalty" : ["l1","l2"]}

logreg_cv = GridSearchCV(pipeline, 
                         grid, 
                         cv = cv)
logreg_cv.fit(X_train,y_train)

best_C = logreg_cv.best_params_['model__C']
best_pen = logreg_cv.best_params_['model__penalty']

print("Best parameters : C = {}, penalty = {} ".format(best_C, best_pen))


# In[11]:


# best_params_LR = pd.DataFrame(data = {'C' : best_C, 
#                                       'Penalty' : best_pen}, 
#                               index = ['LR'])
# best_params_LR.to_csv('best_params_LR.csv')


# ### Model application

# In[50]:


best_params_LR = pd.read_csv('best_params_LR.csv')
best_C = int(best_params_LR['C'][0])
best_pen = best_params_LR['Penalty'][0]
best_params_LR


# In[36]:


# Model with SMOTE method and best parameters

# Model definition
lr = LogisticRegression(max_iter = 100, 
                         C = best_C, 
                         penalty = best_pen)

LR_clf = smote_pip_skfold(X_train, y_train, lr, 'LR_clf')


# In[35]:


y_pred, y_test = evaluate_model(lr)
plot_confusion_matrix(y_pred, y_test)
plt.savefig('cm_lr.png')


# ## Random Forest Classifier with Grid Search

# ### N estimator tuning

# In[38]:


# # Hyper parameters tuning (tree number n)
# min_estimators = 1
# max_estimators = 100
# n_estimators = [i for i in range(min_estimators, max_estimators+1, 10)]
# best_n = pd.DataFrame()

# # Define SMOTE version of the dataset
# oversample_model = SMOTE(sampling_strategy = 'auto', 
#                          k_neighbors = 5)

# # Define a pipeline that first transform training with SMOTE set then fits the model
# forest = RandomForestClassifier(warm_start = False, 
#                                 oob_score = True,
#                                 max_features = 'sqrt', 
#                                 random_state = rs)
# steps = [('oversample', oversample_model), 
#          ('model', forest)]
# pipeline = Pipeline(steps = steps)


# # OOb error as a function of tree number n
# oob_error = pd.DataFrame(columns = ['n','oob'])    
# for i in n_estimators:
#     forest.set_params(n_estimators = i)
#     #forest.fit(X_train, y_train)
#     pipeline.fit(X_train, y_train)
#     d = {'n' : i,'oob' : 1 - forest.oob_score_}
#     oob_error = oob_error.append(d, ignore_index = True)
        
# best_n = best_n.append(oob_error.max(), ignore_index = True)

# plt.figure(figsize = (12,10))
# ax = plt.axes()
# ax.set_ylim(0, 1)
            
# plt.plot(oob_error['n'], oob_error['oob'])
# plt.title(('oob error '), fontsize = 20)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("n_estimators", fontsize = 20)
# plt.ylabel("OOB error rate", fontsize = 20)
# plt.show()

# plt.savefig('oob_error.png')


# ### Hyper parameter tuning

# In[40]:


# # Hyper parameters tuning with Grid Search & SMOTE method

# # Define SMOTE version of the dataset
# oversample_model = SMOTE(sampling_strategy = 'auto', 
#                          k_neighbors = 5)

# # Define a pipeline that first transform training with SMOTE set then fits the model
# RFC = RandomForestClassifier(max_features = 'sqrt', 
#                              oob_score = True, 
#                              random_state = rs)
# steps = [('oversample', oversample_model), 
#          ('model', RFC)]
# pipeline = Pipeline(steps = steps)

# n_var = X_train.shape[1]
# best_params = pd.DataFrame()

# # Define cross validation method
# cv = RepeatedStratifiedKFold(n_splits = 5, 
#                              n_repeats = 3, 
#                              random_state = rs)

# # Define Grid search method
# param_grid = {
#     'model__n_estimators' : [5, 10, 30, 60, 80],
#     'model__min_samples_leaf' : [5, 10, 50],
#     'model__max_depth': [int(n_var/2), int(n_var/3), int(n_var/4)]
# }
# RFC_cv = GridSearchCV(pipeline, 
#                       param_grid = param_grid,
#                       cv = cv)
# RFC_cv.fit(X_train, y_train)

# best_n = RFC_cv.best_params_['model__n_estimators']
# best_msl = RFC_cv.best_params_['model__min_samples_leaf']
# best_md = RFC_cv.best_params_['model__max_depth']

# print("Best parameters : n = {}, min_samples_leaf = {}, max_depth = {} ".format(best_n, best_msl, best_md))


# In[41]:


# best_params_RF = pd.DataFrame(data = {'n' : best_n, 
#                                      'min_samples_leaf' : best_msl,
#                                      'max_depth' : best_md}, 
#                               index = ['RF'])
# best_params_RF.to_csv('best_params_RF.csv')


# ### Model application

# In[45]:


best_params_RF = pd.read_csv('best_params_RF.csv')
best_n = int(best_params_RF['n'][0])
best_msl = best_params_RF['min_samples_leaf'][0]
best_md = best_params_RF['max_depth'][0]
best_params_RF


# In[49]:


# Model definition
RF = RandomForestClassifier(max_depth = best_md,
                            min_samples_leaf = best_msl,
                            n_estimators = best_n,
                            random_state = rs)

RF_clf = smote_pip_skfold(X_train, y_train, RF, 'RF_clf')


# ## XGBoost

# ### Hyper parameters tuning

# In[ ]:


# Hyper parameters tuning with Grid Search & SMOTE method

# Define SMOTE version of the dataset
oversample_model = SMOTE(sampling_strategy = 'auto', 
                         k_neighbors = 5)

# Define a pipeline that first transform training with SMOTE set then fits the model
steps = [('oversample', oversample_model), 
         ('model', XGBClassifier())]
pipeline = Pipeline(steps = steps)

# Define cross validation method
cv = RepeatedStratifiedKFold(n_splits = 5, 
                             n_repeats = 1, 
                             random_state = rs)

# Define Grid search method
grid = {
        'model__min_child_weights': [1, 5, 10],
        'n_estimators' = [10, 50, 100]
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'model__subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
        'model__max_depth': [3, 4, 5]
       }

xgb_cv = GridSearchCV(pipeline, 
                      grid, 
                      cv = cv)

xgb_cv.fit(X_train,y_train)

best_mcw = xgb_cv.best_params_['model__min_child_weights']
best_subsamp = xgb_cv.best_params_['model__subsample']
best_md = xgb_cv.best_params_['model__max_depth']

print("Best parameters : min_child_weights = {}, subsample = {}, max_depth = {} ".format(
    best_mcw, best_subsamp, best_md))


# In[ ]:


best_params_xgb = pd.DataFrame(data = {}, 
                              index = ['XGB'])
best_params_xgb.to_csv('best_params_RF.csv')


# ### Model Application

# In[ ]:


best_params_xgb = pd.read_csv('best_params_xgb.csv')


best_params_xgb


# In[48]:


# Model definition

# xgb = XGBClassifier(
#     n_estimators = best_n
#     min_child_weights = best_mcw,
#     subsample = best_subsamp,
#     max_depth = best_md
# )

xgb = XGBClassifier()
    
xgb_clf = smote_pip_skfold(X_train, y_train, xgb, 'LGBM_clf')


# # Conclusion

# In[47]:


# clf_results = pd.concat([dummy_clf, LR_clf, RF_clf], axis = 0)
# clf_results.to_csv('results_clf.csv')

clf_results = pd.read_csv('results_clf.csv', index_col = 'Unnamed: 0')
clf_results.style.highlight_max(color = 'red', axis = 0).set_precision(2)


# In[ ]:




