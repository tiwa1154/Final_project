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
from copy import deepcopy
import os
import numpy as np
import pandas as pd
# import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import palettes

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


# <<<<<<< HEAD:Final_Project_Tianyi.py

# %%
# Matiral status vs. No loan period
no_loan = df_sub.no_loan
plt.plot(status, no_loan,'o', markersize=3, alpha = 0.1)
plt.ylabel('Month with no loan')
plt.xlabel('Marital status')
plt.title('Matiral status vs. No loan period')
plt.show()
# %%
kids = df_sub.CNT_CHILDREN
plt.plot(kids, debt_sum,'o', markersize=3, alpha = 0.1)
plt.ylabel('Past due period')
plt.xlabel('Number of kids')
plt.title('Matiral status vs. Debt overdue period')
plt.show()
# more people have no kids have longer debt overdue time

# %%
# df_sub.plot(x=kids, y=debt_sum, kind="bar")

#%%
df_sub['OCCUPATION_TYPE'].isna().sum()
# ['OCCUPATION_TYPE'].isna().sum()

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
# a = [11, 12, 13, 14, 15, 16, 17, 18, 19]
# df_m.drop(df_r.columns[a], axis = 1, inplace = True)
# df_m.head()

#%%

df_r.columns
# df_m.credit.values

#%%
#################################################
#################################################
#################################################
#################################################



#%%
g = sns.catplot(x="CODE_GENDER", y="DAYS_BIRTH", hue="credit", palette='mako',
            data=df_m, kind="violin", split=True)
            # .set_title("Age by Gender and Credit User Type")
g.fig.suptitle("Age by Gender and Credit User Type",
                  fontsize=12, fontdict={"weight": "bold"})

#%%
g = sns.catplot(x="CODE_GENDER", y="AMT_INCOME_TOTAL", hue="credit", palette='mako',
            data=df_m, kind="violin", split=True)
            # .set_title("Age by Gender and Credit User Type")
g.fig.suptitle("Income by Gender and Credit User Type",
                  fontsize=12, fontdict={"weight": "bold"})

#%%
g = sns.catplot(x="CODE_GENDER", y="DAYS_EMPLOYED", hue="credit", palette='mako',
            data=df_m, kind="violin", split=True)
            # .set_title("Age by Gender and Credit User Type")
g.fig.suptitle("Days Employed by Gender and Credit User Type",
                  fontsize=12, fontdict={"weight": "bold"})

#%%
g = sns.catplot(x="CODE_GENDER", y="CNT_CHILDREN", hue="credit", palette='mako',
            data=df_m, kind="violin", split=True)

g.fig.suptitle("Count Children by Gender and Credit User Type",
                  fontsize=12, fontdict={"weight": "bold"})


#%%
g = sns.scatterplot(data=df_m, x="DAYS_BIRTH", y="AMT_INCOME_TOTAL", hue = "CODE_GENDER", palette = 'mako').set_title("Credit Users by Income, Age, Gender")

#%%
sns.violinplot(x="credit", y="AMT_INCOME_TOTAL", data=df_m, palette='mako', ).set_title("Income by Credit User Type")

#%%
sns.violinplot(x="credit", y="DAYS_BIRTH", data=df_m, palette='mako').set_title("Age by Credit User Type")

#%%
g = sns.catplot(x="CODE_GENDER", y="AMT_INCOME_TOTAL",
                hue="credit", col="FLAG_OWN_REALTY", palette='mako',
                data=df_m, kind="violin", split=True,
                height=4, aspect=.7);

#%%
g = sns.catplot(x="CODE_GENDER", y="AMT_INCOME_TOTAL",
                hue="credit", col="FLAG_OWN_REALTY", palette='mako',
                data=df_m, kind="violin", split=True,
                height=4, aspect=.7);

#%%
#%%
sns.countplot(x = dfs['NAME_HOUSING_TYPE'], hue = df_m['credit'], orient = 'v', palette='mako').tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = dfs['NAME_FAMILY_STATUS'], hue = df_m['credit'], orient = 'v', palette='mako').tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = dfs['NAME_EDUCATION_TYPE'], hue = df_m['credit'], orient = 'v', palette='mako').tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = dfs['OCCUPATION_TYPE'], hue = df_m['credit'], orient = 'v', palette='mako').tick_params(axis='x', rotation=45)

# sns.barplot(x = imports['features'], y = imports['importances'], hue = imports['world'])


#%%
# f, axes = plt.subplots(2, 1, figsize=(7, 10))
sns.violinplot(x="credit", y="AMT_INCOME_TOTAL", data=df_m).set_title("Good")
# sns.violinplot(x="label_name", y="AMT_INCOME_TOTAL", data=bad, ax = axes[0]).set_title("Bad")




















#%%
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
from copy import deepcopy

#%%
# df_m.shape
# (36457, 22)
# df_m.columns
# df_m.columns['OCCUPATION_TYPE'].isna().sum()

#%%
####################################################
# PREP DATA FOR MODEL
####################################################

features = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH','DAYS_EMPLOYED']

y = df_m['credit']
X = df_m[df_m.columns.intersection(features)]
print(X.columns)

#%%
X.shape
# (36457, 11)
y.shape
# (36457,)

#%%
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
X_train.shape
# (29165, 11)
X_test.shape
# (7292, 11)

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range = range(1,20)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\nReady to continue.")

#%%
import matplotlib.pyplot as plt

plt.plot(k_range, scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

#%%
print(scores)
#2

#%%
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X,y)

#%%
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
scores[k] = metrics.accuracy_score(y_test, y_pred)
scores_list.append(metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\nReady to continue.")

#%%
X_test.shape
# (7292, 11)
y_test.shape
(7292,)
y_pred.shape
(7292,)

#%%

#%%
final = pd.DataFrame(X_test, columns = X.columns)
final['credit'] = np.array(y_test.tolist())
final['credit_pred'] = y_pred.tolist()
# final['gender_cat']

#%%
X

#%%
def outcome_bucket(row):
    bucket = 'FP' if row['credit'] == 0 and row['credit_pred'] == 1 else 'TP' if row['credit'] == 1 and row['credit_pred'] == 1 else 'FN' if row['credit'] == 1 and row['credit_pred'] == 0 else 'TN' if row['credit'] == 0 and row['credit_pred'] == 0 else 'unk'
    return bucket

final['outcome'] = final.apply(outcome_bucket, axis=1) 

#%%
def outcome_category(row):
    bucket = 'RIGHT' if row['outcome'] == 'TP' or row['outcome'] == 'TN' else 'WRONG' if row['outcome'] == 'FP' or row['outcome'] == 'FN' else 'unk'
    return bucket

final['outcome_category'] = final.apply(outcome_category, axis=1) 

#%%
def gender(row):
    bucket = 'FEMALE' if row['CODE_GENDER'] > 0 else 'MALE'
    return bucket

final['gender_category'] = final.apply(gender, axis=1) 

#%%
sns.countplot(x = final['outcome'], hue = final['gender_category'], orient = 'v', palette='mako').tick_params(axis='x', rotation=45)

#%%
g = sns.scatterplot(data=final, x="DAYS_BIRTH", y="AMT_INCOME_TOTAL", hue = "outcome_category", palette = 'mako').set_title("Credit Users by Income, Age, Model Outcome")

#%%
sns.violinplot(x="credit", y="AMT_INCOME_TOTAL", data=final, palette='mako', ).set_title("Income by Credit User Type")

#%%
sns.violinplot(x="credit", y="DAYS_BIRTH", data=df_m, palette='mako').set_title("Age by Credit User Type")

#%%
g = sns.catplot(x="outcome_category", y="AMT_INCOME_TOTAL", hue="gender_category", palette='mako',
            data=final, kind="violin", split=True)
            # .set_title("Age by Gender and Credit User Type")
g.fig.suptitle("Income by Gender and Model Outcome",
                  fontsize=12, fontdict={"weight": "bold"})

#%%
g = sns.catplot(x="outcome_category", y="DAYS_BIRTH", hue="gender_category", palette='mako',
            data=final, kind="violin", split=True)
            # .set_title("Age by Gender and Credit User Type")
g.fig.suptitle("Age by Gender and Model Outcome",
                  fontsize=12, fontdict={"weight": "bold"})

#%%
g = sns.catplot(x="outcome_category", y="DAYS_EMPLOYED", hue="gender_category", palette='mako',
            data=final, kind="violin", split=True)
            # .set_title("Age by Gender and Credit User Type")
g.fig.suptitle("Days Employed by Gender and Model Outcome",
                  fontsize=12, fontdict={"weight": "bold"})

#%%
g = sns.catplot(x="outcome_category", y="DAYS_EMPLOYED", hue="CODE_GENDER", palette='mako',
            data=final, kind="violin", split=True)
            # .set_title("Age by Gender and Credit User Type")
g.fig.suptitle("Days Employed by Gender and Model Outcome",
                  fontsize=12, fontdict={"weight": "bold"})











