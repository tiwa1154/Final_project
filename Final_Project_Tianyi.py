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
from seaborn.palettes import color_palette
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
filepath = os.path.join(os.getcwd())
app = pd.read_csv(os.path.join(filepath, "application.csv"))
app.head(n = 3)

#%%
filepath2 = os.path.join(os.getcwd())
credit = pd.read_csv(os.path.join(filepath2, "credit.csv"))
credit.head(n = 3)

#%%
# Check the discussion here: https://www.kaggle.com/rikdifos/credit-card-approval-prediction/discussion/119320

# To get the wide table
credit['status'] = np.where((credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 1, 0) # define > 60 days past-due

df = credit.pivot(index = 'ID', columns = 'MONTHS_BALANCE', values = 'STATUS')
#%%
df.head(n = 3)

#%%
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
            if paid_off >= no_loan or paid_off <= no_loan:
                good_or_bad.append('good')
            elif paid_off == 0 and no_loan == 1:
                good_or_bad.append('good')
        
        elif overall_overdues != 0:
            if paid_off > overall_overdues:
                good_or_bad.append('good')
            elif paid_off <= overall_overdues:
                good_or_bad.append('bad')
        
        elif paid_off == 0 and no_loan != 0:
            if overall_overdues <= no_loan or overall_overdues >= no_loan:
                good_or_bad.append('bad')

        else:
            good_or_bad.append('good')
                
        
    return good_or_bad
# good(1) or bad(0) set to 1 is good client b/c the gap between # of overdues and payoff or no loan is significant, otherwise, good(1) or bad(0) == 0

#%%

# Inner Join two data sets with intersection ID.
# df = pd.merge(app, df, how="inner", on=["ID", "ID"])
# df.head(3)

#%%
df['pay_off'] = df[df.iloc[:,1:61] == 'C'].count(axis = 1)
df['overdue_1-29'] = df[df.iloc[:,1:61] == '0'].count(axis = 1)
df['overdue_30-59'] = df[df.iloc[:,1:61] == '1'].count(axis = 1)
df['overdue_60-89'] = df[df.iloc[:,1:61] == '2'].count(axis = 1)
df['overdue_90-119'] = df[df.iloc[:,1:61] == '3'].count(axis = 1)
df['overdue_120-149'] = df[df.iloc[:,1:61] == '4'].count(axis = 1)
df['overdue_over_150'] = df[df.iloc[:,1:61] == '5'].count(axis = 1)
df['no_loan'] = df[df.iloc[:,1:61] == 'X'].count(axis = 1)
df['ID'] = df.index
#%%
df.head(10)

#%%
target = pd.DataFrame()
target['ID'] = df.index
target['pay_off'] = df['pay_off'].values
target['#_of_overdues'] = df['overdue_1-29'].values+ df['overdue_30-59'].values + df['overdue_60-89'].values +df['overdue_90-119'].values+df['overdue_120-149'].values +df['overdue_over_150'].values
target['no_loan'] = df['no_loan'].values
target['good(1) or bad(0)'] = feature_engineering_goodbad(df)
df_merge = cleansed_app.merge(target, how="inner", on="ID")
df_merge.head(10)
# df_merge.describe()

#%%
df_merge = df_merge.dropna()

#%%
df_merge.head(5)

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
# finding relationships between good(1) or bad(0)(good/bad) and other features
df_merge2 = df_merge[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'pay_off', '#_of_overdues',
       'no_loan', 'good(1) or bad(0)','OCCUPATION_TYPE','CNT_CHILDREN']]
# Visualize the data using seaborn Pairplots
g = sns.pairplot(df_merge2, hue = 'good(1) or bad(0)', diag_kws={'bw': 0.2})
# observe good relationship between total income and pay off, days employed and pay off, age and no loan/pay off, higher income, less # of overdue; older people with no kids, fewer overdue

#%%
# Investigate all the features by our y
features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'pay_off', '#_of_overdues',
       'good(1) or bad(0)','CNT_CHILDREN','OCCUPATION_TYPE']


for f in features:
    plt.figure()
    ax = sns.countplot(x=f, data=df_merge2, hue = 'good(1) or bad(0)', palette="Set1")
# More famle than male in general, more applicants own properties and good applicants have fewer property counts. more applicants fall under working catagory, have secondary degree, married, have 2 children. good candidates have higher pay off time and usually pay of within a month. Bad candidates have higher # of overdue.

#%%
df_merge2.head()

#%%
# convert catagorical variables to numeric
# numerical_df_merge2 = pd.get_dummies(df_merge2, columns = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','OCCUPATION_TYPE'])
# numerical_df_merge2.head()

#%%
# Scaling columns between 0 and 1 for faster training
# scale_vars = ['AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED','OCCUPATION_TYPE']
# scaler = MinMaxScaler()
# df_merge2[scale_vars] = scaler.fit_transform(df_merge2[scale_vars])
# df_merge2.head()

#%%
df_mergeNew = df_merge2.drop(columns=['pay_off','#_of_overdues','no_loan'],axis=1)
#%%
df_mergeNew.head()

#%%
sns.heatmap(df_mergeNew.corr(), annot=True)
plt.show()
#%%
# Split data
X = df_mergeNew.drop(['good(1) or bad(0)'], axis=1)
y = df_mergeNew['good(1) or bad(0)'].values 
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=1)

#%%
# Decision tree
from sklearn.metrics import classification_report
dt = DecisionTreeClassifier(criterion='entropy',   random_state=1)
dt.fit(X_train, y_train)
print(f'DecisionTreeClassifier train score: {dt.score(X_train,y_train)}')
print(f'DecisionTreeClassifier test score:  {dt.score(X_test,y_test)}')
print(confusion_matrix(y_test, dt.predict(X_test)))
print(classification_report(y_test, dt.predict(X_test))) 

#%%
from sklearn.tree import export_graphviz
import graphviz

dot_data = tree.export_graphviz(dt, out_file=None, 
    feature_names=df_mergeNew.drop(['good(1) or bad(0)'], axis=1).columns,    
    class_names=df_mergeNew['good(1) or bad(0)'].unique().astype(str),  
    filled=True, rounded=True,  
    special_characters=True)
graph = graphviz.Source(dot_data)
graph

#%%
# Calculating FI
for i, column in enumerate(df_mergeNew.drop(['good(1) or bad(0)'], axis=1)):
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
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_pred = dt.predict(X_train)

#%%
# Plotting Confusion Matrix
cm = confusion_matrix(y_train, y_pred)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=dt.classes_, title='Training confusion on good or bad credit')

#%%
# Random Forest
rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)

# source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Accuracy on Test
print("Training Accuracy is: ", rf.score(X_train, y_train))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_test, y_test))
print(confusion_matrix(y_test, rf.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test))) 
# Confusion Matrix
cm = confusion_matrix(y_test, prediction_test)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)

