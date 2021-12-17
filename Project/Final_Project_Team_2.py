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
## Preprocessing
# In preprocessing part, we will clean the data set, making sure that they are in the right shape(eg. join data sets, remove(replace) NA values, 
# correct data type, etc)
 
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
credit = pd.read_csv(os.path.join(filepath, "credit.csv"))
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
##############################
#Functions for data cleaning #
##############################
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
    if m == "Single / not married" or m == "Separated" or m == "Widow":
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
    if edu == "Secondary / secondary special" or edu == "Lower secondary":
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

#%%
#%%
# For Tian's models. (Comprehensive)
def data_cleansing(data):
    # Adding number of family members with number of children to get overall family members.
    data['CNT_FAM_MEMBERS'] = data['CNT_FAM_MEMBERS'] + data['CNT_CHILDREN']
    dropped_cols = ['FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
       'FLAG_EMAIL']
    data = data.drop(dropped_cols, axis = 1)

    data['DAYS_BIRTH'] = np.abs(data['DAYS_BIRTH']/365)
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED']/365 

    housing_type = {'House / apartment' : '1',
                   'With parents': '6',
                    'Municipal apartment' : '2',
                    'Rented apartment': '3',
                    'Office apartment': '4',
                    'Co-op apartment': '5'}
              
    income_type = {'Commercial associate':'1',
                  'State servant':'1',
                  'Working':'1',
                  'Pensioner':'2',
                  'Student':'3'}
                  # 3 major categories, working, student, pensioner
    education_type = {'Secondary / secondary special':'1',
                     'Lower secondary':'1',
                     'Higher education':'2',
                     'Incomplete higher':'2',
                     'Academic degree':'3'}
    family_status = {'Single / not married':'1',
                     'Separated':'2',
                     'Widow':'3',
                     'Civil marriage':'4',
                    'Married':'5'}
    OCCUPATION_TYPE = {'Accountants':'1',
                     'Cleaning staff':'2',
                     'Cooking staff':'3',
                     'Core staff':'4',
                     'Drivers':'5',
                    'HR staff': '6', 
                    'High skill tech staff': '7',
                    'IT staff': '8',
                    'Laborers': '9',
                    'Low-skill Laborers':'10', 
                    'Managers':'11', 
                    'Medicine staff':'12',
                    'Private service staff':'13',
                    'Realty agents':'14', 
                    'Sales staff':'15',
                    'Secretaries':'16',
                    'Security staff':'17',
                    'Waiters/barmen staff':'18'}
    code_gender = {'M':'1',
                     'F':'2'}
    FLAG_OWN_CAR = {'Y':'1',
                    'N':'2'}
    FLAG_OWN_REALTY = {'Y':'1',
                    'N':'2'}
    data['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE'].map(housing_type)
    data['NAME_INCOME_TYPE'] = data['NAME_INCOME_TYPE'].map(income_type)
    data['NAME_EDUCATION_TYPE']=data['NAME_EDUCATION_TYPE'].map(education_type)
    data['NAME_FAMILY_STATUS']=data['NAME_FAMILY_STATUS'].map(family_status)
    data['OCCUPATION_TYPE']=data['OCCUPATION_TYPE'].map(OCCUPATION_TYPE)
    data['CODE_GENDER']=data['CODE_GENDER'].map(code_gender)
    data['FLAG_OWN_CAR']=data['FLAG_OWN_CAR'].map(FLAG_OWN_CAR)
    data['FLAG_OWN_REALTY']=data['FLAG_OWN_REALTY'].map(FLAG_OWN_REALTY)
    return data

cleansed_app = data_cleansing(app)

#%%
def feature_engineering_goodbad(data):
    good_or_bad = []
    for index, row in data.iterrows():
        paid_off = row['pay_off']
        over_1 = row['overdue_1-29']
        over_30 = row['overdue_30-59']
        over_60 = row['overdue_60-89'] 
        over_90 = row['overdue_90-119']
        over_120 = row['overdue_120-149'] + row['overdue_over_150']
        no_loan = row['no_loan']
            
        overall_overdues = over_1+over_30+over_60+over_90+over_120    
            
        if overall_overdues == 0:
            # if paid_off >= no_loan or paid_off <= no_loan:
            #     good_or_bad.append('good')
            if paid_off >= 0 or no_loan >= 0:
                good_or_bad.append('good')
        
        elif overall_overdues != 0:
            if paid_off / overall_overdues >= 1.5:
                good_or_bad.append('good')
            else:
                good_or_bad.append('bad')
        
        elif paid_off == 0 and no_loan != 0:
            if overall_overdues > 0:
                good_or_bad.append('bad')

        else:
            good_or_bad.append('bad')
                
        
    return good_or_bad
#%%
df_merge = dfs.copy()

df_merge["sum_overdue"] = (dfs["overdue_1-29"] + dfs["overdue_30-59"] 
                         + dfs["overdue_60-89"] 
                         + dfs["overdue_90-119"]
                         + dfs["overdue_120-149"]
                         + dfs["overdue_over_150"])
#%%
df_merge = data_cleansing(df_merge)
df_merge['credit'] = feature_engineering_goodbad(df)
df_merge = df_merge.dropna()
df_merge.head(5)

#%%[markdown] 
## EDA
# In EDA part, a detailed summary of the data will be presented, along with 
# graphs and test, preparing for the later model building proccess.

#%%
df_merge.credit.str.get_dummies().sum().plot.pie(label='taget', autopct='%1.0f%%')
# sns.countplot(x="credit", data = df_merge )
# %%
#%%
# Investigate all the elements whithin each Feature 

for column in df_merge:
    unique_vals = np.unique(df_merge[column])
    nr_values = len(unique_vals)
    if nr_values < 20:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))

#%%
# Checking for null values
df_merge.isnull().sum()

#%%
df_merge.columns

#%%
# finding relationships between credit(good/bad) and other features
df_merge2 = df_merge[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'credit','OCCUPATION_TYPE','CNT_CHILDREN']]
# Visualize the data using seaborn Pairplots
g = sns.pairplot(df_merge2, hue = 'credit', diag_kws={'bw': 0.2})
# observe good relationship between total income and pay off, days employed and pay off, age and no loan/pay off, higher income, less 
# of overdue; older people with no kids, fewer overdue

#%%
# Investigate all the features by our y
features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CNT_FAM_MEMBERS',
       'credit','CNT_CHILDREN','OCCUPATION_TYPE']

for f in features:
    plt.figure()
    ax = sns.countplot(x=f, data=df_merge2, hue = 'credit', palette="Set1")
# More famle than male in general, more applicants own properties 
# and good applicants have fewer property counts. more applicants 
# fall under working catagory, have secondary degree, married, have # 2 children. good candidates have higher pay off time and usually 
# pay of within a month. Bad candidates have higher 
# of overdue.
#%%
sns.heatmap(df_merge2.corr(), annot=True)
plt.show()
#%%[markdown]
## Model Building
# Desicion Tree
#%%
df_merge2.head()
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
#%%
#%%
# Split data
X = df_merge2.drop(['credit'], axis=1)
y = df_merge2['credit'].values 
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=1)

#%%
train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 30)]
# evaluate a decision tree for each depth
for i in values:
	# configure the model
	model = DecisionTreeClassifier(max_depth=i)
	# fit model on the training dataset
	model.fit(X_train, y_train)
	# evaluate on the train dataset
	train_yhat = model.predict(X_train)
	train_acc = accuracy_score(y_train, train_yhat)
	train_scores.append(train_acc)
	# evaluate on the test dataset
	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
	# summarize progress
	print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
# plot of train and test scores vs tree depth
plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.legend()
plt.show()

#%%
# Decision tree
from sklearn.metrics import classification_report
dt = DecisionTreeClassifier(criterion='entropy',   random_state=1)
dt.fit(X_train, y_train)
target_names = ['Bad credit', 'Good credit']
print(f'DecisionTreeClassifier train score: {dt.score(X_train,y_train)}')
print(f'DecisionTreeClassifier test score:  {dt.score(X_test,y_test)}')
print(confusion_matrix(y_test, dt.predict(X_test)))
print(classification_report(y_test, dt.predict(X_test),target_names=target_names)) 

#%%
from sklearn.tree import export_graphviz
import graphviz

dot_data = tree.export_graphviz(dt, out_file=None, 
    feature_names=df_merge2.drop(['credit'], axis=1).columns,    
    class_names=df_merge2['credit'].unique().astype(str),  
    filled=True, rounded=True,  
    special_characters=True)
graph = graphviz.Source(dot_data)
graph

#%%
# Calculating FI
for i, column in enumerate(df_merge2.drop(['credit'], axis=1)):
    print('Importance of feature {}:, {:.3f}'.format(column, dt.feature_importances_[i]))
    
    fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [dt.feature_importances_[i]]})
    
    try:
        final_fi = pd.concat([final_fi,fi], ignore_index = True)
    except:
        final_fi = fi
        
        
# Ordering the data
final_fi = final_fi.sort_values('Feature Importance Score', ascending = False).reset_index()            
final_fi

#%%
# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=target_names, yticklabels=target_names, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_pred = dt.predict(X_train)

#%%
# Plotting Confusion Matrix

cm = confusion_matrix(y_train, y_pred)
cm_norm = cm/cm.sum(axis=1)
plt.figure()
plot_confusion_matrix(cm_norm, classes=dt.classes_, title='Training confusion on good or bad credit')
#%%[markdown]
# Random Forest

#%%

rf = RandomForestClassifier(n_estimators=200, criterion='entropy',min_samples_split = 5, min_samples_leaf = 2)
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)

# source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Accuracy on Test
print("Training Accuracy is: ", rf.score(X_train, y_train))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_test, y_test))
print(confusion_matrix(y_test, rf.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test),target_names=target_names)) 
# Confusion Matrix
cm = confusion_matrix(y_test, prediction_test)
cm_norm = cm/cm.sum(axis=1)
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)


# %%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE 

# %%
rf_cv_acc = cross_val_score(rf, X_train, y_train, cv= 10, scoring='accuracy', n_jobs=-1 )
print(f'LR CV accuracy score:',  rf_cv_acc.mean())
# %%
dt_cv_acc = cross_val_score(dt, X_train, y_train, cv= 10, scoring='accuracy', n_jobs=-1 )
print(f'LR CV accuracy score:',  dt_cv_acc.mean())
# %%
