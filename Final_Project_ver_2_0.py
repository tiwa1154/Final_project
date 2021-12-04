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
# * [GitHub Link](https://github.com/tiwa1154/Final_project)

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
# * DAYS_BIRTH	Birthday:	 
#                            Count backwards from current day (0),  -1 means yesterday
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
# * STATUS:	           0: 1-29 days past due 1: 30-59 days past due 2: 60-89 days overdue 3: 90-119 days overdue 4: 120-149 days overdue 5: Overdue or bad debts, write-offs for more than 150 days C: paid off that month X: No loan for the month

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
filepath = os.path.join(os.getcwd())
app = pd.read_csv(os.path.join(filepath, "application.csv"))
app.head(n = 3)

#%%
filepath2 = os.path.join(os.getcwd())
credit = pd.read_csv(os.path.join(filepath2, "credit.csv"))
credit.head(n = 3)

#%%
# Check the discussion here: https://www.kaggle.com/rikdifos/credit-card-approval-y_test_pred/discussion/119320

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
# dfs.to_csv('subset.csv')
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
df_sub = dfs.copy()
#%%
# Make sum of the total loan based on subset of dataframe, easier for visualization
df_sub['sum_column']= df_sub.iloc[:,20:26].sum(axis=1)
df_sub.head()
# higher sum means more days overdue

#%%
# total annual income vs. debt overdue period
fuzzyincome = df_sub.AMT_INCOME_TOTAL + np.random.normal(0,1, size=len(df_sub.AMT_INCOME_TOTAL))
debt_sum = df_sub.sum_column + np.random.normal(0,1, size=len(df_sub.sum_column))
plt.plot(fuzzyincome, debt_sum,'o', markersize=3, alpha = 0.1)
# sns.boxplot(y="sum_column", x="AMT_INCOME_TOTAL",
#               color="b",
#                data=df_sub)
plt.ylabel('Past due summary')
plt.xlabel('Annual income')
plt.title('Annual income vs. Debt overdue period ')
plt.show()
# higher income, less overdue

# %%
# marital status vs debt overdue period
status = df_sub.NAME_FAMILY_STATUS
plt.plot(status, debt_sum,'o', markersize=3, alpha = 0.1)
plt.ylabel('Past due period')
plt.xlabel('Marital status')
plt.title('Matiral status vs. Debt overdue period')
plt.show()
# Married population has a longer debt overdue period compare to other marital status

#%%
print(df_sub.NAME_INCOME_TYPE.value_counts())

# %%
# add work type
sns.scatterplot(x=status, y=debt_sum, hue="NAME_INCOME_TYPE", data=df_sub)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Past due period')
plt.xlabel('Marital status')
plt.title('Matiral status vs. Debt overdue period')
plt.show()
# conclusion?
#%%
# There are might be some EDA that still needs to be done.
# But we'll see. 
#######################################################
# The following procedures are for the model building #
#######################################################
#%% 
# df = dfs.copy() 
# use this to create deep copy of the data frame if other algothrithm
# needs specific variables. 

#%%
# Drop the unwanted columns.
df_r = dfs.copy()
df_r = df_r.drop(['ID','FLAG_MOBIL', 'FLAG_WORK_PHONE', 
                      'FLAG_PHONE', 'FLAG_EMAIL',
                      'OCCUPATION_TYPE'], 
                       axis = 1)
# %%
# Convert gender into dummy variables
# Male = 0    Female = 1
def gender_convert(GENDER):
    if GENDER == 'M':
        return 0
    else: 
        return 1
#%%
df_r["CODE_GENDER"] = df_r["CODE_GENDER"].apply(gender_convert)
# %%
# Convert N & Y to dummy variables
def ny_convert(ny):
    if ny == "N":
        return 0
    else:
        return 1
# It is possible to combine gender_convert and ny_convert. 
# But I choose not to.
#%%
df_r["FLAG_OWN_CAR"] = df_r["FLAG_OWN_CAR"].apply(ny_convert)
df_r["FLAG_OWN_REALTY"] = df_r["FLAG_OWN_REALTY"].apply(ny_convert)
#%%
# For convenience, it is easy to sort marital this way. 
# No partenership: 0, else 1.
def marr_convert(m):
    if m == "Single / not married" or "Separated" or "Widow":
        return 0
    else:
        return 1
#%%
df_r["NAME_FAMILY_STATUS"] = df_r["NAME_FAMILY_STATUS"].apply(ny_convert)
# %%
# Same way for housing type
def house_convert(h):
    if h == "With parents":
        return 0
    else:
        return 1
#%%
df_r["NAME_HOUSING_TYPE"] = df_r["NAME_HOUSING_TYPE"].apply(house_convert)
# %%
def income_convert(income):
    if income == "Student":
        return 0
    elif income == "Pensioner":
        return 1
    else:
        return 2
#%%
df_r["NAME_INCOME_TYPE"] = df_r["NAME_INCOME_TYPE"].apply(income_convert)
# %%
def edu_convert(edu):
    if edu == "Secondary / secondary special" or "Lower secondary":
        return 0
    elif edu == "Higher education" or "Incomplete higher":
        return 1
    else:
        return 2
#%%
df_r["NAME_EDUCATION_TYPE"] = df_r["NAME_EDUCATION_TYPE"].apply(edu_convert)

# %%
# Now, I would like to convert DAYS_BIRTH into their actual age.
df_r[["DAYS_BIRTH"]] = df_r[["DAYS_BIRTH"]].apply(lambda x: abs(x)/365, axis = 1)
# %%
# Same as DATS_EMPLOYED
df_r[["DAYS_EMPLOYED"]] = df_r[["DAYS_EMPLOYED"]].apply(lambda x: abs(x)/365, axis = 1)
# %%
# Last one is to deal with the STATUS
# Get sum of over_due
df_r["sum_overdue"] = (df_r["overdue_1-29"] + df_r["overdue_30-59"] 
                         + df_r["overdue_60-89"] 
                         + df_r["overdue_90-119"]
                         + df_r["overdue_120-149"]
                         + df_r["overdue_over_150"])
 
#%%
df_r.head()
# Alight, no more characters in df_r               	
# %%
# We would like a binary response of good or bad customer as response
# We define 0: Bad Credit, 1: Good Credit
# I am not very familiar with credit system. 
# So, I use this https://www.creditkarma.com/credit-cards/i/late-payments-affect-credit-score 
# and https://www.investopedia.com/ask/answers/021215/what-good-debt-ratio-and-what-bad-debt-ratio.asp 
# as a reference.
# Good Credit: no_load, pay_off, or pay_off > overdue (Since the credit
# score slowly improved while starting paying on time. )
# Bad Credit: pay_off <= overdue
def customer(df):
    credit = []
    for i in range(0, len(df)):
        if df["sum_overdue"][i] == 0: # No Overdue. Well, good.
            if df["no_loan"][i] >=0 or df["pay_off"][i] >= 0:
                credit.append(1)
        elif df["sum_overdue"][i] != 0:
            if df["pay_off"][i]/df["sum_overdue"][i] >= 1.5: # good
                credit.append(1)
            else: # ratio < 1.5 is defined as bad credit
                credit.append(0)
        elif df["pay_off"][i] == 0 and df["no_loan"][i] !=0:
            if df["sum_overdue"][i] >0: # 
                credit.append(0)
            else:
                credit.append(1)
    df["credit"] = credit
    return df
#%%
df_m = customer(df_r)
#%%
a = [11, 12, 13, 14, 15, 16, 17, 18, 19]
df_m.drop(df_r.columns[a], axis = 1, inplace = True)
df_m.head()
# %%
# ! pip install xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from xgboost import plot_tree
#%%
x = df_m.drop(['credit'], axis=1)
y = df_m['credit']
#%%
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=1)
#%%
xgb_fit = XGBClassifier(max_depth = 10)
xgb_fit.fit(X_train, y_train)
print('XGBoost Model Accuracy : ', xgb_fit.score(X_test, y_test)*100, '%')

y_test_pred = xgb_fit.predict(X_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, y_test_pred))
      
print('\nClassification report:')      
print(classification_report(y_test, y_test_pred))

# %%
# Feature importance (XGBoost)
pd.DataFrame({'Variable':X_train.columns,'Importance':
        xgb_fit.feature_importances_}).sort_values('Importance', ascending=False)
# %%
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
fig, ax = plt.subplots(figsize=(200, 250))
plot_tree(xgb_fit, ax=ax, rankdir='LR')
plt.show()
# %%
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lr_fit = LogisticRegression()
lr_fit.fit(X_train,y_train)
#%%
print('Logistic Regression Model Accuracy : ', lr_fit.score(X_test, y_test)*100, '%')

y_test_pred2 = lr_fit.predict(X_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, y_test_pred2))
      
print('\nClassification report:')      
print(classification_report(y_test, y_test_pred2))
#%%
# ! pip install scikit-plot
import scikitplot as skplt
y_pred_proba = lr_fit.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_pred_proba)
plt.show()
# %%
