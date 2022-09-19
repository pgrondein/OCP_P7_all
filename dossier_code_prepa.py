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
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

import warnings
warnings.simplefilter(action ='ignore', category = FutureWarning) 


# # Data 

# In[2]:


path = 'C:/Users/pgron/Jupyter/P7/data/'
rs = 42

# Main table, broken into two files for Train (with TARGET) and Test (without TARGET).
app_test = pd.read_csv(path + 'application_test.csv', nrows = None) #not used
app_train = pd.read_csv(path + 'application_train.csv', nrows = None)
appli_train, appli_test = train_test_split(app_train, test_size = 0.20)

# All client's previous credits provided by other financial institutions that were reported to Credit Bureau
# For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau 
# before the application date.
df_bureau = pd.read_csv(path + 'bureau.csv', nrows = None)
bureau_train, bureau_test = train_test_split(df_bureau, test_size = 0.20, random_state = rs)

# Monthly balances of previous credits in Credit Bureau.
bb = pd.read_csv(path + 'bureau_balance.csv', nrows = None)
bb_train, bb_test = train_test_split(bb, test_size = 0.20, random_state = rs)

# Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit
df_pos_cash = pd.read_csv(path + 'POS_CASH_balance.csv', nrows = None)
pos_train, pos_test = train_test_split(df_pos_cash, test_size = 0.20, random_state = rs)

# Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
credit_card = pd.read_csv(path + 'credit_card_balance.csv', nrows = None)
cc_train, cc_test = train_test_split(credit_card, test_size = 0.20, random_state = rs)

# All previous applications for Home Credit loans of clients who have loans in our sample.
prev_app = pd.read_csv(path + 'previous_application.csv', nrows = None)
prev_train, prev_test = train_test_split(prev_app, test_size = 0.20, random_state = rs)

# Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
# There is a) one row for every payment that was made plus b) one row each for missed payment.
inst_pay = pd.read_csv(path + 'installments_payments.csv', nrows = None)
inst_train, inst_test = train_test_split(inst_pay, test_size = 0.20, random_state = rs)


# In[3]:


# This file contains descriptions for the columns in the various data files.
desc_col = pd.read_csv(path + 'HomeCredit_columns_description.csv', nrows = None, encoding= 'unicode_escape')
desc_col.head(5)


# # Preparation 

# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# 
# The following key ideas were used:
#  - Divide or subtract important features to get rates (like annuity and income)
#  - In Bureau Data: create specific features for Active credits and Closed credits
#  - In Previous Applications: create specific features for Approved and Refused applications
#  - Modularity: one function for each table (except bureau_balance and application_test)
#  - One-hot encoding for categorical features
# 
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
#  You can use LightGBM with KFold or Stratified KFold.
# 
# Update 16/06/2018:
#  - Added Payment Rate feature
#  - Removed index from features
#  - Use standard KFold CV (not stratified)

# ## Functions 

# In[4]:


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# In[5]:


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    #Convert categorical variable into dummy/indicator variables
    df = pd.get_dummies(df, columns = categorical_columns, dummy_na = nan_as_category)
    
    new_columns = [c for c in df.columns if c not in original_columns]
    
    return df, new_columns


# In[6]:


# Preprocess application_train.csv and application_test.csv
def application_train_test(df, num_rows = None, nan_as_category = False):
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
        
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
#     del test_df
    
    gc.collect()
    
    return df


# In[7]:


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(df_1, df_2, num_rows = None, nan_as_category = True):

    bb, bb_cat = one_hot_encoder(df_2, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(df_1, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    
    # For each column in bb_cat, add a 'mean' version of this column in bb_aggregations
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
        
    #display(bb_aggregations)
        
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    
    #display(bb_agg)
    
    bureau = bureau.join(bb_agg, how = 'left', on = 'SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis = 1, inplace = True)
    
    del bb, bb_agg
    #display(df_bureau)
    
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
#     display(bureau_cat)
#     display(bb_cat)
#     display(cat_aggregations)
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
#     display(bureau_agg)
    
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how ='left', on ='SK_ID_CURR')
    
    del active, active_agg
    
    gc.collect()
    
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    del closed, closed_agg, bureau
    
    gc.collect()
    
    return bureau_agg


# In[8]:


# Preprocess previous_applications.csv
def previous_applications(df, num_rows = None, nan_as_category = True):
    
    prev, cat_cols = one_hot_encoder(df, nan_as_category= True)
    
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    
    # For each column, add a 'mean' version of this column
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    
    prev_agg = prev_agg.join(approved_agg, how ='left', on='SK_ID_CURR')
    
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    
    prev_agg = prev_agg.join(refused_agg, how = 'left', on = 'SK_ID_CURR')
    
    del refused, refused_agg, approved, approved_agg, prev
    
    gc.collect()
    
    return prev_agg


# In[9]:


# Preprocess POS_CASH_balance.csv
def pos_cash(df, num_rows = None, nan_as_category = True):
    
    pos, cat_cols = one_hot_encoder(df, nan_as_category= True)
    
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
    del pos
    
    gc.collect()
    
    return pos_agg


# In[10]:


# Preprocess installments_payments.csv
def installments_payments(df, num_rows = None, nan_as_category = True):

    ins, cat_cols = one_hot_encoder(df, nan_as_category= True)
    
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    
    for cat in cat_cols:
        aggregations[cat] = ['mean']
        
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    del ins
    
    gc.collect()
    
    return ins_agg


# In[11]:


# Preprocess credit_card_balance.csv
def credit_card_balance(df, num_rows = None, nan_as_category = True):
    
    cc, cat_cols = one_hot_encoder(df, nan_as_category= True)
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis = 1, inplace = True)
    
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    
    del cc
    
    gc.collect()
    
    return cc_agg


# In[12]:


# Join all df together after aggregation
def main(debug = False):
    
    num_rows = 10000 if debug else None
    
    df = application_train_test(app_train, app_test, num_rows)
    
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(df_bureau, bb, num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how = 'left', on = 'SK_ID_CURR')
        #del bureau
        gc.collect()
        
    with timer("Process previous_applications"):
        prev = previous_applications(prev_app, num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how ='left', on = 'SK_ID_CURR')
        #del prev
        gc.collect()
        
    with timer("Process POS-CASH balance"):
        pos = pos_cash(df_pos_cash, num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how = 'left', on = 'SK_ID_CURR')
        #del pos
        gc.collect()
        
    with timer("Process installments payments"):
        ins = installments_payments(inst_pay, num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how = 'left', on = 'SK_ID_CURR')
        #del ins
        gc.collect()
        
    with timer("Process credit card balance"):
        cc = credit_card_balance(credit_card, num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how = 'left', on = 'SK_ID_CURR')
        #del cc
        gc.collect()
        
#     with timer("Run LightGBM with kfold"):
#         feat_importance = kfold_lightgbm(df, num_folds= 10, stratified= False, debug= debug)

    return df


# In[13]:


# Join all df together after aggregation
def main_train_test(debug = False):
    
    num_rows = 10000 if debug else None
    
    df_train = application_train_test(appli_train, num_rows)
    df_test = application_train_test(appli_test, num_rows)
    
    with timer("Process bureau and bureau_balance"):
        bur_train = bureau_and_balance(bureau_train, bb_train, num_rows)
        bur_test = bureau_and_balance(bureau_test, bb_test, num_rows)
        
        df_train = df_train.join(bur_train, how = 'left', on = 'SK_ID_CURR')
        df_test = df_test.join(bur_test, how = 'left', on = 'SK_ID_CURR')

        del bur_train, bur_test
        gc.collect()
        
    with timer("Process previous_applications"):
        prev_app_train = previous_applications(prev_train, num_rows)
        prev_app_test = previous_applications(prev_test, num_rows)

        df_train = df_train.join(prev_app_train, how = 'left', on = 'SK_ID_CURR')
        df_test = df_test.join(prev_app_test, how = 'left', on = 'SK_ID_CURR')

        del prev_app_train, prev_app_test
        gc.collect()
        
    with timer("Process POS-CASH balance"):
        pos_cash_train = pos_cash(pos_train, num_rows)
        pos_cash_test = pos_cash(pos_train, num_rows)

        df_train = df_train.join(pos_cash_train, how = 'left', on = 'SK_ID_CURR')
        df_test = df_test.join(pos_cash_test, how = 'left', on = 'SK_ID_CURR')

        del pos_cash_train, pos_cash_test
        gc.collect()
        
    with timer("Process installments payments"):
        ins_train = installments_payments(inst_train, num_rows)
        ins_test = installments_payments(inst_test, num_rows)

        df_train = df_train.join(ins_train, how = 'left', on = 'SK_ID_CURR')
        df_test = df_test.join(ins_test, how = 'left', on = 'SK_ID_CURR')

        del ins_train, ins_test
        gc.collect()
        
    with timer("Process credit card balance"):
        ccb_train = credit_card_balance(cc_train, num_rows)
        ccb_test = credit_card_balance(cc_test, num_rows)

        df_train = df_train.join(ccb_train, how = 'left', on = 'SK_ID_CURR')
        df_test = df_test.join(ccb_test, how = 'left', on = 'SK_ID_CURR')

        del ccb_train, ccb_test
        gc.collect()
        
    return df_train, df_test


# In[14]:


def def_pd(df, df_name) : #function to explore dataset
    
    #STRUCTURE
    display(df.info())
    #print('--- Unique values ---')
    #display(df.nunique())
    
    #NaN
    #Calcul du pct de valeurs manquantes total
    row = df.shape[0]
    col = df.shape[1]
    nb_val = row * col
    nb_val_manqu = df.isnull().sum().sum()
    val_manq_pct = round((nb_val_manqu/nb_val)*100, 3)
    
    print('--- Nan & Duplicated ---')
    print('   ')
    print('Le fichier {} comporte {} lignes et {} colonnes,' 
          ' ainsi que {} valeurs manquantes sur {} entrées ({} %).'.format(df_name, row,
                                                                           col, nb_val_manqu, nb_val, val_manq_pct))
    #Calcul du pourcentage de valeurs manquantes par variables
    #index = df.index
    nb_na_var = df.isnull().mean()
    pct_remplissage_var = pd.DataFrame((100 - (nb_na_var*100)).astype(int), 
                                        columns = ['Pourcentage de remplissage (%)'])
    #print('   ')
    #display(pct_remplissage_var)
    #print('   ')

    #Calcul du pourcentage de valeurs manquantes par individus
    nb_na_ind = df.isnull().mean(axis = 1)
    pct_remplissage_ind = pd.DataFrame((100 - (nb_na_ind*100)).astype(int), 
                                        columns = ['Pourcentage de remplissage (%)'])   
    #print('   ')
    #display(pct_remplissage_ind)
    #print('   ')

    
    #DOUBLONS
    dup = df.duplicated().sum()
    print('   ')
    print('Le dataframe comporte', dup, 'doublons globaux.')
    print('   ')
    
#     for col in df.columns :
#         n = df.duplicated(subset = col).sum()
#         print(f'col : {col} -> duplicated : {n}')
    
    
#    display(df.describe(include = np.number))
#    display(df.describe(include = object))

    return pct_remplissage_var, pct_remplissage_ind


# In[15]:


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


# ## Function application 

# In[16]:


df_train, df_test = main_train_test()
print(df_train.shape)
print(df_test.shape)


# In[17]:


list_col_train = df_train.columns
list_col_test = df_test.columns
drop_list = []

for col in list_col_train : 
    if col not in list_col_test : 
        drop_list.append(col)
        
drop_list


# In[19]:


df_train = df_train.drop(drop_list, axis = 1)
df_train.shape


# In[20]:


df = pd.concat([df_train, df_test], axis = 0)
pie(df, 'TARGET', lim = 0)


# # Traitement données manquantes

# In[21]:


# Check if there's infinite values
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
print(np.all(np.isfinite(df_train[feats])))
print(np.all(np.isfinite(df_test[feats])))


# In[22]:


# Replace infinite values with nan values
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_test = df_test.replace([np.inf, -np.inf], np.nan)


# In[23]:


# Train/Test details
pct_var_train, pct_ind_train = def_pd(df_train[feats], 'Train')
pct_var_test, pct_ind_test = def_pd(df_test[feats], 'Test')


# In[24]:


lim_var = 50

# On sépare les variables remplies des autres
var_keep_train = pct_var_train.loc[pct_var_train['Pourcentage de remplissage (%)'] > lim_var]
var_drop_train = pct_var_train.loc[pct_var_train['Pourcentage de remplissage (%)'] < lim_var]

# var_keep_test = pct_var_test.loc[pct_var_test['Pourcentage de remplissage (%)'] > lim_var]
# var_drop_test = pct_var_test.loc[pct_var_test['Pourcentage de remplissage (%)'] < lim_var]

pct_keep_train = round((len(var_keep_train.index)/len(df_train.columns))*100, 2)
pct_drop_train = round((len(var_drop_train.index)/len(df_train.columns))*100, 2)

# pct_keep_test = round((len(var_keep_test.index)/len(df_test.columns))*100, 2)
# pct_drop_test = round((len(var_drop_test.index)/len(df_test.columns))*100, 2)

print('TRAIN')
print('On garde ', len(var_keep_train.index), ' variables ({} %).'.format(pct_keep_train))
print('On retire ', len(var_drop_train.index), ' variables ({} %).'.format(pct_drop_train))
print('  ')
# print('TEST')
# print('On garde ', len(var_keep_test.index), ' variables ({} %).'.format(pct_keep_test))
# print('On retire ', len(var_drop_test.index), ' variables ({} %).'.format(pct_drop_test))


# In[25]:


# On retire les variables trop peut remplies dans train/test
df_train = df_train.drop(var_drop_train.index, axis = 1)
df_test = df_test.drop(var_drop_train.index, axis = 1)


# In[26]:


# On traite les valeurs manquantes des variables restantes 
feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

for col in feats :
    med_train = df_train[col].median()
    df_train[col] = df_train[col].fillna(med_train)
    
    df_test[col] = df_test[col].fillna(med_train)
    
def_pd(df_train[feats], 'Train')
def_pd(df_test[feats], 'Test')


# In[27]:


scaler = MinMaxScaler()
df_train = pd.DataFrame(data = scaler.fit_transform(df_train), 
                        columns = df_train.columns, index = df_train.index)

df_test = pd.DataFrame(data = scaler.transform(df_test), 
                        columns = df_test.columns, index = df_test.index)


# In[28]:


# Infinite values check
print(np.all(np.isfinite(df_train[feats])))
print(np.all(np.isfinite(df_test[feats])))

#NaN values check
print(np.all(np.isnan(df_train[feats])))
print(np.all(np.isnan(df_test[feats])))


# In[29]:


df_train.to_csv('df_train.csv')
df_test.to_csv('df_test.csv')


# In[30]:


df_train.he


# In[ ]:





# In[ ]:




