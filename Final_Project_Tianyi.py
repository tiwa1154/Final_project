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

    housing_type = {'House / apartment' : 'House / apartment',
                   'With parents': 'With parents',
                    'Municipal apartment' : 'House / apartment',
                    'Rented apartment': 'House / apartment',
                    'Office apartment': 'House / apartment',
                    'Co-op apartment': 'House / apartment'}
              
    income_type = {'Commercial associate':'Working',
                  'State servant':'Working',
                  'Working':'Working',
                  'Pensioner':'Pensioner',
                  'Student':'Student'}
    education_type = {'Secondary / secondary special':'secondary',
                     'Lower secondary':'secondary',
                     'Higher education':'Higher education',
                     'Incomplete higher':'Higher education',
                     'Academic degree':'Academic degree'}
    family_status = {'Single / not married':'Single',
                     'Separated':'Single',
                     'Widow':'Single',
                     'Civil marriage':'Married',
                    'Married':'Married'}
    data['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE'].map(housing_type)
    data['NAME_INCOME_TYPE'] = data['NAME_INCOME_TYPE'].map(income_type)
    data['NAME_EDUCATION_TYPE']=data['NAME_EDUCATION_TYPE'].map(education_type)
    data['NAME_FAMILY_STATUS']=data['NAME_FAMILY_STATUS'].map(family_status)
    return data

cleansed_app = data_cleansing(app)

#%%
def feature_engineering_target(data):
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
                good_or_bad.append(1)
            elif paid_off == 0 and no_loan == 1:
                good_or_bad.append(1)
        
        elif overall_overdues != 0:
            if paid_off > overall_overdues:
                good_or_bad.append(1)
            elif paid_off <= overall_overdues:
                good_or_bad.append(0)
        
        elif paid_off == 0 and no_loan != 0:
            if overall_overdues <= no_loan or overall_overdues >= no_loan:
                good_or_bad.append(0)

        else:
            good_or_bad.append(1)
                
        
    return good_or_bad
# target set to 1 is good client b/c the gap between # of overdues and payoff or no loan is significant, otherwise, target == 0

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
target['target'] = feature_engineering_target(df)
df_merge = cleansed_app.merge(target, how="inner", on="ID")
df_merge.head(10)
df_merge.describe()

#%%
df_merge.O

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
# finding relationships between target(good/bad) and other features
df_merge2 = df_merge[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'pay_off', '#_of_overdues',
       'no_loan', 'target','OCCUPATION_TYPE']]
# Visualize the data using seaborn Pairplots
g = sns.pairplot(df_merge2, hue = 'target', diag_kws={'bw': 0.2})
# observe good relationship between total income and pay off, days employed and pay off, age and no loan/pay off, higher income, less # of overdue; older people with no kids, fewer overdue
#%%
# kids vs features
df_merge2 = df_merge[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', '#_of_overdues',
       'CNT_CHILDREN','OCCUPATION_TYPE']]
# Visualize the data using seaborn Pairplots
g = sns.pairplot(df_merge2, hue = '#_of_overdues', diag_kws={'bw': 0.2}, palette = "tab10")


#%%
# occupation type vs features
df_merge2 = df_merge[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'pay_off', '#_of_overdues',
       'target','CNT_CHILDREN','OCCUPATION_TYPE']]
# Visualize the data using seaborn Pairplots
g = sns.pairplot(df_merge2, hue = 'OCCUPATION_TYPE', diag_kws={'bw': 0.2}, palette = "tab10")
#%%
# Investigate all the features by our y
features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'pay_off', '#_of_overdues',
       'target','CNT_CHILDREN','OCCUPATION_TYPE']


for f in features:
    plt.figure()
    ax = sns.countplot(x=f, data=df_merge2, hue = 'target', palette="Set1")
# More famle than male in general, more applicants own properties and good applicants have fewer property counts. more applicants fall under working catagory, have secondary degree, married, have 2 children. good candidates have higher pay off time and usually pay of within a month. Bad candidates have higher # of overdue.

#%%
df_merge2.head()

#%%
# convert catagorical variables to numeric
numerical_df_merge2 = pd.get_dummies(df_merge2, columns = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','OCCUPATION_TYPE'])
numerical_df_merge2.head()

#%%
# Scaling columns between 0 and 1 for faster training
scale_vars = ['AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED']
scaler = MinMaxScaler()
numerical_df_merge2[scale_vars] = scaler.fit_transform(numerical_df_merge2[scale_vars])
numerical_df_merge2.head()

#%%
# Split data
X = numerical_df_merge2.drop(['#_of_overdues','pay_off', 'no_loan'], axis=1).values
y = numerical_df_merge2['#_of_overdues'].values 
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=0)

#%%
# Decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
dt.fit(X_train, y_train)

#%%
from sklearn.tree import export_graphviz
import graphviz

dot_data = tree.export_graphviz(dt, out_file=None, 
    feature_names=numerical_df_merge2.drop(['#_of_overdues','pay_off', 'no_loan'], axis=1).columns,    
    class_names=numerical_df_merge2['#_of_overdues'].unique().astype(str),  
    filled=True, rounded=True,  
    special_characters=True)
graph = graphviz.Source(dot_data)
graph

#%%
# Calculating FI
for i, column in enumerate(numerical_df_merge2.drop(['#_of_overdues','pay_off', 'no_loan'], axis=1)):
    print('Importance of feature {}:, {:.3f}'.format(column, dt.feature_importances_[i]))
    
    fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [dt.feature_importances_[i]]})
    
    try:
        final_fi = pd.concat([final_fi,fi], ignore_index = True)
    except:
        final_fi = fi
        
        
# Ordering the data
final_fi = final_fi.sort_values('Feature Importance Score', ascending = False).reset_index()            
final_fi
# the only features effect the score are # of overdues, pay_off, NAME_FAMILY_STATUS_Widow, FLAG_OWN_CAR_N, no_loan, AMT_INCOME_TOTAL

#%%
# Split data
X = numerical_df_merge2.drop(['#_of_overdues','pay_off','target', 'no_loan'], axis=1).values
y = numerical_df_merge2['#_of_overdues'].values 
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=0)

#%%
# Decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
dt.fit(X_train, y_train)

#%%
# if drop target column, #_of_overdues is based on age and emplyment period most heavily.
from sklearn.tree import export_graphviz
import graphviz

dot_data = tree.export_graphviz(dt, out_file=None, 
    feature_names=numerical_df_merge2.drop(['#_of_overdues','pay_off', 'target', 'no_loan'], axis=1).columns,    
    class_names=numerical_df_merge2['#_of_overdues'].unique().astype(str),  
    filled=True, rounded=True,  
    special_characters=True)
graph = graphviz.Source(dot_data)
graph

#%%
# Split data
X = numerical_df_merge2.drop('target', axis=1).values
y = numerical_df_merge2['target'].values 
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=0)

#%%
# Decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
dt.fit(X_train, y_train)

#%%
# if drop target column, #_of_overdues is based on age and emplyment period most heavily.
from sklearn.tree import export_graphviz
import graphviz

dot_data = tree.export_graphviz(dt, out_file=None, 
    feature_names=numerical_df_merge2.drop('target', axis=1).columns,    
    class_names=numerical_df_merge2['target'].unique().astype(str),  
    filled=True, rounded=True,  
    special_characters=True)
graph = graphviz.Source(dot_data)
graph
#%%
# Accuracy on Train
print("Training Accuracy is: ", dt.score(X_train, y_train))

# Accuracy on Train
print("Testing Accuracy is: ", dt.score(X_test, y_test))

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
plot_confusion_matrix(cm_norm, classes=dt.classes_, title='Training confusion')

#%%
# Random Forest


rf = RandomForestClassifier(n_estimators=25134, criterion='entropy')
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)

# source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Accuracy on Test
print("Training Accuracy is: ", rf.score(X_train, y_train))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_test, y_test))

# Confusion Matrix
cm = confusion_matrix(y_test, prediction_test)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)

#%%


rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)

# source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Accuracy on Test
print("Training Accuracy is: ", rf.score(X_train, y_train))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_test, y_test))

# Confusion Matrix
cm = confusion_matrix(y_test, prediction_test)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)


#%%
# xtarget = df_merge[df_merge.drop('target',axis = 1).columns]
# ytarget = df_merge['target']
# xtrain, xtest, ytrain, ytest = train_test_split(xtarget,ytarget, train_size = 0.8,stratify=ytarget, random_state = 0)

#%%
# from sklearn.tree import DecisionTreeClassifier
# # Import train_test_split
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix 
# from sklearn.metrics import classification_report
# # Instantiate dtree
# # dtree_admit1 = DecisionTreeClassifier(max_depth=2, random_state=1)
# # # Fit dt to the training set
# # dtree_admit1.fit(xtrain,ytrain)
# # # Predict test set labels
# # y_test_pred = dtree_admit1.predict(xtest)
# # # Evaluate test-set accuracy
# # print(accuracy_score(ytest, y_test_pred))
# # print(confusion_matrix(ytest, y_test_pred))
# # print(classification_report(ytest, y_test_pred))
# # # %%
# # # Subset dataframe 
# # dfs = df.iloc[:, np.r_[0:1, 61:87]]
# # dfs.head(10)
# # test classification dataset
# from sklearn.datasets import make_classification
# # define dataset
# X, y = make_classification(n_samples=36457, n_features=20, n_informative=8, n_redundant=5, random_state=3)
# # summarize the dataset
# print(X.shape, y.shape)

# #%%
# from numpy import mean
# from numpy import std
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# # define the model
# model = RandomForestClassifier()
# # evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# # # Check the data type of each column. 
# # dfs.iloc[:,0:26].info(verbose=True)

# #%%
# def get_dataset():
# 	X, y = make_classification(n_samples=36457, n_features=20, n_informative=8, n_redundant=5, random_state=3)
# 	return X, y
 
# # get a list of models to evaluate
# def get_models():
# 	models = dict()
# 	# explore number of features from 1 to 7
# 	for i in range(1,8):
# 		models[str(i)] = RandomForestClassifier(max_features=i)
# 	return models
 
# # evaluate a given model using cross-validation
# def evaluate_model(model, X, y):
# 	# define the evaluation procedure
# 	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 	# evaluate the model and collect the results
# 	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# 	return scores
 
# # define dataset
# X, y = get_dataset()
# # get the models to evaluate
# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
# 	# evaluate the model
# 	scores = evaluate_model(model, X, y)
# 	# store the results
# 	results.append(scores)
# 	names.append(name)
# 	# summarize the performance along the way
# 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# # plot model performance for comparison
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()
# # %%
# # Write a summary function to have a glance of the numeric part
# # of the data set
# def var_summary(x):
#     return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(), x.std(), x.var(), x.min(), 
#         x.quantile(0.01), x.quantile(0.05), x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), 
#                               x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 
                     
#                   index = ['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1', 
#                                'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])
# #%%
# # Select the numeric variables from app. 
# print("Summary of Numeric Varible")
# df_nu = df_merge.select_dtypes([np.number]) 
# df_nu.apply(var_summary).T
# Everything looks fine. 

#%%
# Delete this when you start coding and analyzing:
# I have set up the data set, along with the summary of all the variables. We need some plots of the data for visulization next. 

#%%
# Make sum of the total loan based on subset of dataframe, easier for visualization
# df_sub = pd.read_csv('subset.csv')
# df_sub['sum_column']= df_sub.iloc[:,20:26].sum(axis=1)
# df_sub.head()
# higher sum means more days overdue

# #%%
# df_sub.groupby('OCCUPATION_TYPE')['OCCUPATION_TYPE'].count()

# #%%
# df_sub_null = df_sub.dropna(axis=0)
# df_sub_null.isnull().any()

# # #%%
# # df_sub_clean['FLAG_OWN_CAR'] = pd.factorize(df_sub_clean['FLAG_OWN_CAR'])[0]
# # df_sub_clean['FLAG_OWN_REALTY'] = pd.factorize(df_sub_clean['FLAG_OWN_REALTY'])[0]
# # df_sub_clean['NAME_INCOME_TYPE'] = pd.factorize(df_sub_clean['NAME_INCOME_TYPE'])[0]
# # df_sub_clean['NAME_EDUCATION_TYPE'] = pd.factorize(df_sub_clean['NAME_EDUCATION_TYPE'])[0]
# # df_sub_clean['NAME_FAMILY_STATUS'] = pd.factorize(df_sub_clean['NAME_FAMILY_STATUS'])[0]
# # df_sub_clean['NAME_HOUSING_TYPE'] = pd.factorize(df_sub_clean['NAME_HOUSING_TYPE'])[0]
# # df_sub_clean['OCCUPATION_TYPE'] = pd.factorize(df_sub_clean['OCCUPATION_TYPE'])[0]
# # df_sub_clean.dtypes

# #%%
# # total annual income vs. debt overdue period
# fuzzyincome = df_sub.AMT_INCOME_TOTAL + np.random.normal(0,1, size=len(df_sub.AMT_INCOME_TOTAL))
# debt_sum = df_sub.sum_column + np.random.normal(0,1, size=len(df_sub.sum_column))
# plt.plot(fuzzyincome, debt_sum,'o', markersize=3, alpha = 0.1)
# # sns.boxplot(y="sum_column", x="AMT_INCOME_TOTAL",
# #               color="b",
# #                data=df_sub)
# plt.ylabel('Past due summary')
# plt.xlabel('Annual income')
# plt.title('Annual income vs. Debt overdue period ')
# plt.show()
# # higher income, less overdue

# # %%
# # marital status vs debt overdue period
# status = df_sub.NAME_FAMILY_STATUS
# plt.plot(status, debt_sum,'o', markersize=3, alpha = 0.1)
# plt.ylabel('Past due period')
# plt.xlabel('Marital status')
# plt.title('Matiral status vs. Debt overdue period')
# plt.show()
# # Married population has a longer debt overdue period compare to other marital status

# #%%
# print(df_sub.NAME_INCOME_TYPE.value_counts())

# # %%
# # add work type
# sns.scatterplot(x=status, y=debt_sum, hue="NAME_INCOME_TYPE", data=df_sub)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.ylabel('Past due period')
# plt.xlabel('Marital status')
# plt.title('Matiral status vs. Debt overdue period')
# plt.show()
# # conclusion?

# # %%
# # Matiral status vs. No loan period
# no_loan = df_sub.no_loan
# plt.plot(status, no_loan,'o', markersize=3, alpha = 0.1)
# plt.ylabel('Month with no loan')
# plt.xlabel('Marital status')
# plt.title('Matiral status vs. No loan period')
# plt.show()
# # %%
# kids = df_sub.CNT_CHILDREN
# plt.plot(kids, debt_sum,'o', markersize=3, alpha = 0.1)
# plt.ylabel('Past due period')
# plt.xlabel('Number of kids')
# plt.title('Matiral status vs. Debt overdue period')
# plt.show()
# # more people have no kids have longer debt overdue time

# # %%
# # df_sub.plot(x=kids, y=debt_sum, kind="bar")
# # plt.show()


# #%%
# cleansed_df_sub = data_cleansing(df_sub)

# #%%
# df_sub.NAME_FAMILY_STATUS.value_counts()
# #%%
# def cleandf_subFamilyStatus(row, colname): 
#   thisamt = row[colname]
#   if (thisamt == "Married"): return "1"
#   if (thisamt == "Single / not married"): return "2"
#   if (thisamt == "Civil marriage"): return "3" 
#   if (thisamt == "Separated"): return "4"
#   if (thisamt == "Widow"): return "5"
  
#%%



# # %%
# # clustering
# from numpy import where
# from sklearn.datasets import make_classification
# from matplotlib import pyplot
# # define dataset
# X, y = make_classification(n_samples=36457, n_subnumerical_df_merge2=9, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=10)
# # create scatter plot for samples from each class
# for class_value in range(2):
# 	# get row indexes for samples with this class
# 	row_ix = where(y == class_value)
# 	# create scatter of these samples
# 	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# # show the plot
# pyplot.show()

# %%
