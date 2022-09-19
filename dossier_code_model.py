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
import shap

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action = 'ignore')


# # Data 

# In[2]:


rs = 42

# Séparation train/test
train_df = pd.read_csv('df_train.csv').drop('Unnamed: 0', axis = 1)
test_df = pd.read_csv('df_test.csv').drop('Unnamed: 0', axis = 1)

feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

X_train = train_df[feats]
y_train = train_df['TARGET']

X_test = test_df[feats]
y_test = test_df['TARGET']


# ## Functions

# In[3]:


def display_importances(feature_importance_df_, model):
    
    cols = feature_importance_df_[
           ["feature", "importance"]].groupby("feature").mean().sort_values(
           by = "importance", ascending = False)[:40].index
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize = (8, 10))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", 
                                                                                  ascending = False))
    plt.title('Features Importance')
    plt.tight_layout()
    plt.savefig('{}_importances.png'.format(model))


# In[34]:


def model_app_train(X_train, y_train, model, model_name, grid, class_imb = 'CW', ybot = 0, ytop = 1, 
                    n_fold = 5, n_fold_repeats = 5) : 
     
    
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
    
    
    # CROSS VALIDATION METHOD
    cv = RepeatedStratifiedKFold(n_splits = n_fold, 
                                 n_repeats = n_fold_repeats, 
                                 random_state = rs)
    
    model_cv = GridSearchCV(pipeline, 
                            grid, 
                            cv = cv,
                            scoring = 'roc_auc',
                            refit = True)
    best_model = model_cv.fit(X_train,y_train)
        
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
              
    boxplot(auc, 'Fold', 'AUC', ybot, ytop)

    if model_name == 'Dummy_clf' :
        return best_model
    if model_name == 'LR_clf' :
        coef = model_cv.best_estimator_.named_steps.model.coef_
        feature_importance_df_ = pd.DataFrame(data = {'feature' : feats, 'importance' : coef[0]})
        display_importances(feature_importance_df_ , model_name)
        return best_model
    if model_name == 'RF_clf' :
        coef = model_cv.best_estimator_.named_steps.model.feature_importance_
        feature_importance_df_ = pd.DataFrame(data = {'feature' : feats, 'importance' : coef})
        display_importances(feature_importance_df_ , model_name)
        return best_model


# In[5]:


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

#     # Plot normalized confusion matrix
#     plt.figure(figsize = (8,8))
#     generate_confusion_matrix(cnf_matrix, 
#                               classes = class_names, 
#                               normalize = True, 
#                               title = 'Normalized confusion matrix')
#     plt.show()
        
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


# In[17]:


def boxplot(df, x_, y_, ybot, ytop):
    
    boxprops = dict(linestyle = '-', linewidth = 1, color = 'k')
#     medianprops = dict(linestyle = '-', linewidth = 1, color = 'k')
#     meanprops = dict(marker = 'D', markeredgecolor = 'black', markerfacecolor = 'firebrick')
    sns.set_style("whitegrid")

    plt.figure(figsize = (10, 8))

    sns.boxplot(x = x_, 
                y = y_, 
                data = df, 
                boxprops = boxprops, 
                showfliers = True, 
#                 medianprops = medianprops, 
                showmeans = True, 
#                 meanprops = meanprops
               )

    plt.title('AUC', fontsize = 30)
    plt.ylim(ybot, ytop)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Fold', fontsize = 25)
    plt.ylabel('AUC', fontsize = 25)
    plt.show()


# In[29]:


def predict(model, model_name, X_test, y_test, average_recall = 'binary', average_prec = 'binary'):
    
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]
    
    auc = roc_auc_score(y_test, y_score)
#     print(auc)
    f1 = f1_score(y_test, y_pred)
#     print(f1)
    recall = recall_score(y_test, y_pred, average = average_recall)
#     print(recall)
    prec = precision_score(y_test, y_pred, average = average_prec)
#     print(prec)
    
    predict = pd.DataFrame({'AUC' : round(auc, 2), 
                            'F1' : round(f1, 2), 
                            'Recall score' : round(recall, 2), 
                            'Precision' : round(prec, 2)}, 
                           index = [model_name])
    display(predict)
    
    plot_confusion_matrix(y_pred, y_test)
    
    return predict


# # Models

# ## Dummy Classifier

# In[35]:


# Model definition
dum = DummyClassifier(strategy = "most_frequent")
grid_dum = {}

best_dum = model_app_train(X_train, y_train, dum, 'Dummy_clf', grid = grid_dum, class_imb = 'CW', ybot = 0, ytop = 1)


# In[36]:


pred_dum = predict(best_dum, 'Dummy_clf', X_test, y_test, average_recall = 'weighted', average_prec = 'weighted')
pred_dum.to_csv('dum_score.csv')


# ## Logistic Regression

# In[37]:


logreg = LogisticRegression(class_weight = 'balanced')

grid_lr = {"model__C" : [0.001, 0.01, 0.1, 1, 10, 100], 
        "model__penalty" : ["l1","l2"]}

logreg_cv = model_app_train(X_train, y_train, logreg, 'LR_clf', grid = grid_lr, class_imb = 'CW', ybot = 0, ytop = 1,
                           n_fold = 5, n_fold_repeats = 1)


# In[38]:


pred_logreg = predict(logreg_cv, 'LR_clf', X_test, y_test)
pred_logreg.to_csv('LR_score.csv')


# ## Random Forest Classifier 

# In[56]:


# Hyper parameters tuning (tree number n)
min_estimators = 1
max_estimators = 100
n_estimators = [i for i in range(min_estimators, max_estimators+1, 10)]
best_n = pd.DataFrame()

# Define a pipeline that first transform training with SMOTE set then fits the model
forest = RandomForestClassifier(warm_start = False, 
                                oob_score = True,
                                max_features = 'sqrt', 
                                random_state = rs,
                                class_weight = 'balanced')

# OOb error as a function of tree number n
oob_error = pd.DataFrame(columns = ['n','oob'])    
for i in n_estimators:
    forest.set_params(n_estimators = i)
    forest.fit(X_train, y_train)
    d = {'n' : i,'oob' : 1 - forest.oob_score_}
    oob_error = oob_error.append(d, ignore_index = True)
        
best_n = best_n.append(oob_error.max(), ignore_index = True)

plt.figure(figsize = (12,10))
ax = plt.axes()
ax.set_ylim(0, 1)
            
plt.plot(oob_error['n'], oob_error['oob'])
plt.title(('oob error '), fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("n_estimators", fontsize = 20)
plt.ylabel("OOB error rate", fontsize = 20)
plt.show()

plt.savefig('oob_error.png')
plt.show()


# In[ ]:


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

best_RF = model_app_train(X_train, y_train, forest, 'RF_clf', grid = grid_RF, class_imb = 'CW', ybot = 0, ytop = 1,
                         n_fold = 5, n_fold_repeats = 1)


# In[ ]:


pred_RF = predict(best_RF, 'RF_clf', X_test, y_test)
pred_RF.to_csv('RF_score.csv')


# # Conclusion

# In[ ]:


clf_results = pd.concat([pred_dum, pred_LR, pred_RF], axis = 0)
clf_results.to_csv('results_clf.csv')

# clf_results = pd.read_csv('results_clf.csv', index_col = 'Unnamed: 0')
clf_results.style.highlight_max(color = 'red', axis = 0).set_precision(2)


# In[ ]:




