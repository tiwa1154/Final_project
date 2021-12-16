#%%
# Sharing some hot keys to help with editing. Happy Coding, team.
#
# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# To undo 'ctrl + Z'
# To wrap for long text 'alt + Z'
# To edit multiple line 'ctrl + alt + arrow'
# To comment out code 'ctrl + /'
# To edit text with the same content 'ctrl + shift + L'
#%%[markdown]
## DAT - 6103 - Data Mining - Final Project
#
# Team 2 Members: 
# *  Tianyi Wang
# *  Baldur Hua
# *  Chiemeziem Oguayo
# *  Meg Elliott Fitzgerald
# %%[markdown]
## Introduction
# The average American is over $5,000 in credit card debt, a figure that’s back on the rise now that household spending has recovered from the 2019 pandemic-induced lull. This exceeds the average amount of credit card debt in any other country, and will likely only increase as times goes on if current trends continue. With so many of us frequently relying on credit cards, it behooves us to understand the patterns and profiles associated with credit card usage. Armed with this knowledge, we are able to recognize when certain characteristics make individuals more likely to accrue large balances or default on payments. For instance, based on data collected by Shift, on average men, individuals between the ages of 45-54, Generation X’ers, and those making more than $160,000 per year have the highest amount of credit card debt. By analyzing our dataset we hope to develop multi-dimensional profiles of credit card customers, and then explore whether these profiles are helpful in predicting credit card usage and payment behavior. In short, we hope to answer the following questions:
#
#   1. Can we unveil the relationship between individuals’ financial condition and their personal condition?
# 
#   2. Most importantly, can we develop customer profiles using clustering based on applicant data in order to predict future account behavior at the end?
# 
# * [Data Source](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)
# 
# * [GitHub Link](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)

#%%[markdown]
# Data Description (Application.csv):
# * ID:	Client               number	
# * CODE_GENDER:	         Gender	
# * FLAG_OWN_CAR:	         Is there a car	
# * FLAG_OWN_REALTY:	     Is there a property	
# * CNT_CHILDREN:	         Number of children	
# * AMT_INCOME_TOTAL:	     Annual income	
# * NAME_INCOME_TYPE:	     Income category	
# * NAME_EDUCATION_TYPE:	 Education level	
# * NAME_FAMILY_STATUS:	     Marital status	
# * NAME_HOUSING_TYPE:	     Way of living	
# * DAYS_BIRTH	Birthday:	 Count backwards from current day (0),  -1 means yesterday
# * DAYS_EMPLOYED:	         Start date of employment	Count backwards from current day(0). If positive, it means the person currently unemployed.
# * FLAG_MOBIL:	             Is there a mobile phone	
# * FLAG_WORK_PHONE:	     Is there a work phone	
# * FLAG_PHONE:	             Is there a phone	
# * FLAG_EMAIL:	             Is there an email	
# * OCCUPATION_TYPE:	     Occupation	
# * CNT_FAM_MEMBERS:	     Family size	
# 
# Data Decription (Credit.csv)	
# * ID:	               Client number	
# * MONTHS_BALANCE:	   The month of the extracted data is the starting point, backwards, 0 is the current month, -1 is the previous month, and so on
# * STATUS:	            0: 1-29 days past due 
#                       1: 30-59 days past due 
#                       2: 60-89 days overdue 
#                       3: 90-119 days overdue 
#                       4: 120-149 days overdue 
#                       5: Overdue or bad debts, write-offs for more than 150 days 
#                       C: paid off that month 
#                       X: No loan for the month


#%%
df.columns
#%%
df_nu


# %%[markdown]
## Preprocessing & EDA
# In preprocessing part, we will clean the data set, making sure that they are in the right shape(eg. join data sets, remove(replace) NA values, 
# correct data type, etc)
#
# In EDA part, a detailed summary of the data will be presented, along with graphs and test, preparing for the later model building 
# proccess. 
# %%
import os
import numpy as np
import pandas as pd
# import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# read in application dataset
filepath = os.path.join(os.getcwd())
app = pd.read_csv(os.path.join(filepath, "application.csv"))
app.head(n = 3)
#%%
# read in credit dataset
filepath2 = os.path.join(os.getcwd())
credit = pd.read_csv(os.path.join(filepath2, "credit.csv"))
credit.head(n = 3)

#%%
# Check the discussion here: https://www.kaggle.com/rikdifos/credit-card-approval-prediction/discussion/119320

# To get the wide table
credit['status'] = np.where((credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 1, 0) # define > 60 days past-due

credit_piv = credit.pivot(index = 'ID', columns = 'MONTHS_BALANCE', values = 'STATUS')
#%%
credit_piv.head(n = 3)
#%%
# Inner Join two data sets with intersection ID.
df = pd.merge(app, credit_piv, how="inner", on=["ID", "ID"])
df.head(3)
# len(df)
# %%
# Count number of different status.
df['pay_off'] = df[df.iloc[:,18:79] == 'C'].count(axis = 1)
df['overdue_1-29'] = df[df.iloc[:,18:79] == '0'].count(axis = 1)
df['overdue_30-59'] = df[df.iloc[:,18:79] == '1'].count(axis = 1)
df['overdue_60-89'] = df[df.iloc[:,18:79] == '2'].count(axis = 1)
df['overdue_90-119'] = df[df.iloc[:,18:79] == '3'].count(axis = 1)
df['overdue_120-149'] = df[df.iloc[:,18:79] == '4'].count(axis = 1)
df['overdue_over_150'] = df[df.iloc[:,18:79] == '5'].count(axis = 1)
df['no_loan'] = df[df.iloc[:,18:79] == 'X'].count(axis = 1)
#%%
# df.head(3)
# %%
# Subset dataframe 
dfs = df.iloc[:, np.r_[0:18, 79:87]]
dfs.head(3)
# %%
#%%
# Check the data type of each column. 
dfs.iloc[:,0:26].info(verbose=True)
# %%
# Write a summary function to have a glance of the numeric part
# of the data set
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(), x.std(), x.var(), x.min(), 
        x.quantile(0.01), x.quantile(0.05), x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), 
                              x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 
                     
                  index = ['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1', 
                               'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])
#%%
# Select the numeric variables from app. 
print("Summary of Numeric Varible")
df_nu = dfs.select_dtypes([np.number]) 
df_nu.apply(var_summary).T
# Everything looks fine. 
#%%
# Delete this when you start coding and analyzing:
# I have set up the data set, along with the summary of all the variables. We need some plots of the data for visulization next. 


#################################################
#################################################
#################################################
#################################################
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats import weightstats as stests
from sklearn.model_selection import cross_val_score
import warnings

#%%
df.columns = df.columns.str.lower()

#%%
# create target column
# good customer: 
#   never use card (no loan), pay off in full each month (pay off), overdue_1-29
# bad customer:
#   overdue_30-59, overdue_60-89, overdue_90-119, overdue_120-149, overdue_over_150

def classify(row):
    if row['overdue_30-59'] + row['overdue_60-89'] + row['overdue_90-119'] + row['overdue_120-149'] + row['overdue_over_150'] > 0:
        return "bad"
    else:
        return "good"
    
df['target'] = df.apply(classify, axis=1) 

#%%
sns.countplot(x = df['target'], hue = df['code_gender']).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['target'], hue = df['name_housing_type']).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['target'], hue = df['name_income_type']).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['target'], hue = df['name_education_type']).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['occupation_type'], hue = df['target']).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['name_family_status'], hue = df['target']).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['cnt_children'], hue = df['target']).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['cnt_fam_members'], hue = df['target']).tick_params(axis='x', rotation=45)

#%%
sns.displot(df, x="amt_income_total", hue="target", kind="kde", multiple="stack")

#%%
sns.displot(df, x="days_birth", hue="target", kind="kde", multiple="stack")

#%%
sns.displot(df, x="days_employed", hue="target", kind="kde", multiple="stack")

#%%
# change target to binary
df['target_bi'] = df['target'].replace({"good": 1, "bad": 0}, inplace=False)
df['target_bi'] = df['target_bi'].astype(int)
df['target_bi'].unique()

#%%
# name features
features = ['code_gender', 'flag_own_car', 'flag_own_realty', 'cnt_children', 'amt_income_total', 'name_income_type', 'name_education_type', 'name_family_status', 'name_housing_type', 'days_birth', 'days_employed', 'flag_mobil', 'flag_work_phone', 'flag_phone', 'flag_email', 'occupation_type', 'cnt_fam_members']

#%%
####################################################
# PREP DATA FOR MODEL
####################################################

y = df['target_bi']
X = df[df.columns.intersection(features)]
X.columns
# encode x labels
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform)

#%%
# separate test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#%%
####################################################
# FEATURE REDUCTION
####################################################

# create function to run through different n's to see which would be best for PCA
def pca_opt(n_list):
    pca_results_dict = {}
    for i in n_list:
        pca = PCA(n_components=i)
        X_train_pca = pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        pca_results_dict[i] = np.round(np.sum(pca.explained_variance_ratio_), 2)
    # dataframe = pd.DataFrame.from_dict(pca_results_dict)
    return pca_results_dict

# CHOOSE PCA COMPONENTS N
pca_results = pca_opt(range(0,15))
pca_df = pd.DataFrame.from_dict(pca_results, orient = 'index', columns = ['explained_variance'])
e = sns.lineplot(x=pca_df.index.to_list(), y=pca_df["explained_variance"])
e.set(xlabel='PCA n Components', ylabel='Explained Variance')
plt.show()

#%%
# going to go with n = 10 to keep the explained variance around 80-85% to avoid overfitting
pca = PCA(n_components=10)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
final_explained_variance = np.round(np.sum(pca.explained_variance_ratio_), 2)
final_explained_variance

####################################################
# MODEL BUILDING
####################################################
#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) # instantiate with n value given
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)
print(y_pred)
print(knn.score(X_test,y_test))
# 0.897989412184217

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\nReady to continue.")

#%%
# Calculating error for K values between 1 and 40
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

#%%
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=30) # instantiate with n value given
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)
print(y_pred)
print(knn.score(X_test,y_test))
# 0.897989412184217

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\nReady to continue.")


####################################################
####################################################
####################################################
####################################################




#%%
# K-Nearest-Neighbor KNN 
# number of neighbors
mrroger = 7

#%%
# KNN algorithm
# Re-do our darta with scale on X
from sklearn.preprocessing import scale
xs = pd.DataFrame( scale(X), columns=X.columns )  
# Note that scale( ) coerce the object from pd.dataframe to np.array  
# Need to reconstruct the pandas df with column names
xs.rank = X.rank
ys = y.copy()  # no need to scale y, but make a true copy / deep copy to be safe

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn.fit(xs,ys)
y_pred = knn.predict(xs)
y_pred = knn.predict_proba(xs)
print(y_pred)
print(knn.score(xs,ys))
# 0.897989412184217

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ys, y_pred))
print(classification_report(ys, y_pred))

print("\nReady to continue.")

#%%
xs

#%%
# 2-KNN algorithm
# The better way
# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1 )
knn_split = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn_split.fit(x_train,y_train)
ytest_pred = knn_split.predict(x_test)
ytest_pred
print(knn_split.score(x_test,y_test))
# 0.8789906747120132

# Try different n values

print("\nReady to continue.")

#%%
# 3-KNN algorithm
# The best way
knn_cv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given

from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(knn_cv, X, y, cv=10)
print(cv_results) 
print(np.mean(cv_results)) 
# [0.86423478 0.82912781 0.85655513 0.80718596 0.831322   0.81952825
#  0.81239715 0.83840878 0.80521262 0.82359396]
# 0.8287566433177046

print("\nReady to continue.")

#%%
# 4-KNN algorithm
# Scale first? better or not?

# Re-do our darta with scale on X
from sklearn.preprocessing import scale
xs = pd.DataFrame( scale(X), columns=X.columns )  
# Note that scale( ) coerce the object from pd.dataframe to np.array  
# Need to reconstruct the pandas df with column names
xs.rank = X.rank
ys = y.copy()  # no need to scale y, but make a true copy / deep copy to be safe

print("\nReady to continue.")

#%%
# from sklearn.neighbors import KNeighborsClassifier
knn_scv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given

# from sklearn.model_selection import cross_val_score
scv_results = cross_val_score(knn_scv, xs, ys, cv=5)
print(scv_results) 
print(np.mean(scv_results)) 

print("\nReady to continue.")
# [0.85340099 0.84393856 0.83376766 0.82142367 0.83006446]
# 0.8365190690015034



##############################################################################################

# Last one is to deal with the STATUS
# Get sum of over_due
df["sum_overdue"] = (df["overdue_1-29"] + df_r["overdue_30-59"] 
                         + df["overdue_60-89"] 
                         + df["overdue_90-119"]
                         + df["overdue_120-149"]
                         + df_r["overdue_over_150"])








