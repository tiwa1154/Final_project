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

#%%
df.columns = df.columns.str.lower()

#%%
# create target column
# never use card (no loan)
# pay off in full each month (pay off)
# worst is past due (past due)
# worst is overdue (overdue)
def classify(row):
    no_loan = row['no_loan']
    pay_off = row['pay_off']
    pastdue = row['overdue_1-29'] + row['overdue_30-59']
    overdue = row['overdue_60-89'] + row['overdue_90-119'] + row['overdue_120-149'] + row['overdue_over_150']
    total = no_loan + pay_off + pastdue + overdue
    if pay_off == total:
        return 'pay off'
    elif no_loan == total:
        return 'no loan'
    elif pay_off + no_loan == total:
        return 'pay off'
    elif overdue > 0:
        return 'overdue'
    elif pastdue > 0:
        return 'past due'
    else: 
        return 'unk'
    
df['target'] = df.apply(classify, axis=1) 

#%%
# categorical variables to test next to target
# * NAME_INCOME_TYPE:	     Income category	
# * NAME_EDUCATION_TYPE:	 Education level	
# * NAME_FAMILY_STATUS:	     Marital status	
# * NAME_HOUSING_TYPE:	     Way of living
# * OCCUPATION_TYPE:	     Occupation	
# * FLAG_MOBIL:	             Is there a mobile phone	
# * FLAG_WORK_PHONE:	     Is there a work phone	
# * FLAG_PHONE:	             Is there a phone	
# * FLAG_EMAIL:	             Is there an email
# * CODE_GENDER:	         Gender	
# * FLAG_OWN_CAR:	         Is there a car	
# * FLAG_OWN_REALTY:	     Is there a property

# numerical variables to test next to target
# * DAYS_BIRTH	Birthday:	 Count backwards from current day (0),  -1 means yesterday
# * DAYS_EMPLOYED:	         Start date of employment	Count backwards from current day(0). If positive, it means the person currently unemployed.
# * AMT_INCOME_TOTAL:	     Annual income
# * CNT_CHILDREN:	         Number of children	
# * CNT_FAM_MEMBERS:	     Family size

#%%
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

warnings.filterwarnings("ignore")

# SUBROUTINE 1
# use categorical features to determine correlation with overall satisfaction in order to
# gauge how helpful they will be in a model
def category(dataset, feature, target):
    labels = dataset[feature].unique().tolist()
    data_labels = np.array(dataset[feature])
    data_targets = np.array(dataset[target])
    list_names = []
    lists = {}
    means = dataset[[feature, target]].groupby([feature], as_index=False).mean()
    for item in labels:
        list_names.append(item)
        item_targets = [data_targets[x] for x in range(0, len(data_targets)-1) if data_labels[x] == item]
        lists[item] = item_targets
        item_index = means.loc[means[feature] == item].index[0]
        print("Sample is", round((len(item_targets)/len(data_targets)*100)), "% ", item, "with a mean satisfaction of", str(np.round(means.iloc[item_index, 1], 2)), ".")
    if len(labels) == 2:
        ztest, pval = stests.ztest(x1=lists[list_names[0]], x2=lists[list_names[1]], value=0, alternative='two-sided')
        # print(float(pval))
        if pval < 0.05:
            print("{}: {} and {} groups have significantly different satisfaction levels. (p-value is {})".format(feature, list_names[0], list_names[1], str(pval)))
        else:
            print("{}: %s and %s do not have significantly different satisfaction levels. (p-value is {})".format(feature, list_names[0], list_names[1], str(pval)))
    else:
        pass





# Numerical vs Numerical : Use correlation coefficient
# Numerical vs categorical : Use correlation ratio
# Categorical vs categorical : Use Cramer's coefficient.

# Numerical vs categorical:
# A simple approach could be to group the continuous variable using the categorical variable, measure the variance in each group and comparing it to the overall variance of the continuous variable. If the variance after grouping falls down significantly, it means that the categorical variable can explain most of the variance of the continuous variable and so the two variables likely have a strong association. If the variables have no correlation, then the variance in the groups is expected to be similar to the original variance.

#%%
# Categorical vs Categorical:
#load necessary packages and functions
import scipy.stats as stats
import numpy as np

#create 2x2 table
data = np.array([[6,9], [8, 5], [12, 9]])

#Chi-squared test statistic, sample size, and minimum of rows and columns
X2 = stats.chi2_contingency(data, correction=False)[0]
n = np.sum(data)
minDim = min(data.shape)-1

#calculate Cramer's V 
V = np.sqrt((X2/n) / minDim)

#display Cramer's V
print(V)

0.1775

#%%
data = pd.crosstab(index=df['target'], columns=df['name_income_type'])
data.reset_index(inplace = True, drop = True)
# data.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
data
print(data.shape[1])         # Show dimension of thecolumns
print(range(data.shape[1]))   # Show range of the columns

data.columns = range(data.shape[1])   # Delete headers
data
# #Chi-squared test statistic, sample size, and minimum of rows and columns
X2 = stats.chi2_contingency(data, correction=False)[0]
n = np.sum(data)
minDim = min(data.shape)-1
# #calculate Cramer's V 
V = np.sqrt((X2/n) / minDim)
# #display Cramer's V
print(V)



#%%
target_order = ['no loan', 'pay off', 'past due', 'overdue'] 

#%%
sns.countplot(x = df['target'], hue = df['code_gender'], order = target_order).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['target'], hue = df['name_housing_type'], order = target_order).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['target'], hue = df['name_income_type'], order = target_order).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['target'], hue = df['name_education_type'], order = target_order).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['occupation_type'], hue = df['target']).tick_params(axis='x', rotation=45)

#%%
df_features.columns
#%%
# features
features = ['id', 'code_gender', 'flag_own_car', 'flag_own_realty', 'cnt_children', 'amt_income_total', 'name_income_type', 'name_education_type', 'name_family_status', 'name_housing_type', 'days_birth', 'days_employed', 'flag_mobil', 'flag_work_phone', 'flag_phone', 'flag_email', 'occupation_type', 'cnt_fam_members', 'target']
df_features = df[df.columns.intersection(features)]

#%%
#Using Pearson Correlation
plt.figure(figsize=(20,1))
cor = df_features.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Correlation with output variable
cor_target = abs(cor["target"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features

#%%
df_features.dtypes
# need to encode categorical variables (reg or one-hot?)
# need to do pearson's corr matrix - or see if there's another way to gauge corr of categorical variables (cramer's v?)


















































#%%
    
    # total = df.iloc[first:second, [79, 80, 81, 82, 83, 84, 85, 86]].sum(axis=1).astype(int)
    # pay_off = df.iloc[first:second, [79]].sum(axis=1).astype(int)
    # pastdue = df.iloc[first:second, [80, 81]].sum(axis=1).astype(int)
    # overdue = df.iloc[first:second, [82, 83, 84, 85]].sum(axis=1).astype(int)
    # no_loan = df.iloc[first:second, [86]].sum(axis=1).astype(int)
    # if pay_off == total:
    #     return 'pay off'
    # elif no_loan == total:
    #     return 'no loan'
    # elif overdue > 0:
    #     return 'overdue'
    # elif pastdue > 0:
    #     return 'past due'
    # else:
    #     return 'unk'

# df.apply(classify, axis=1)

def classify(row):
    first = int(row)
    second = row+1
    total = df.iloc[first:second, [79, 80, 81, 82, 83, 84, 85, 86]].sum(axis=1).astype(int)
    pay_off = df.iloc[first:second, [79]].sum(axis=1).astype(int)
    pastdue = df.iloc[first:second, [80, 81]].sum(axis=1).astype(int)
    overdue = df.iloc[first:second, [82, 83, 84, 85]].sum(axis=1).astype(int)
    no_loan = df.iloc[first:second, [86]].sum(axis=1).astype(int)
    if pay_off == total:
        return 'pay off'
    elif no_loan == total:
        return 'no loan'
    elif overdue > 0:
        return 'overdue'
    elif pastdue > 0:
        return 'past due'
    else:
        return 'unk'

df.apply(classify, axis=1)
#%% 
df_nu


I’m interested in rows 10 till 25 and columns 3 to 5.

titanic.iloc[9:25, 2:5]





#%%
# categorize features
df.columns= df.columns.str.lower()
df

#%%
f, axes = plt.subplots(1, 2, figsize=(15, 7))
# f.xticks(rotation = 45)
sns.countplot(x = df[(df['code_gender'] == 'M')]['name_income_type'], ax = axes[0]).set_title("M")
sns.countplot(x = df[(df['code_gender'] == 'F')]['name_income_type'], ax = axes[1]).set_title("F")
axes[0].set(xlabel='Male', ylabel='count')
axes[1].set(xlabel='Female', ylabel='count')
axes[0].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['name_income_type'], hue = df['code_gender']).tick_params(axis='x', rotation=45)

#%%
sns.countplot(x = df['code_gender'], hue = df['name_income_type']).tick_params(axis='x', rotation=45)






#%%
flags = ['flag_own_car', 'flag_own_realty', 'flag_mobil', 'flag_work_phone', 'flag_phone', 'flag_email']
categoricals = ['code_gender', 'name_income_type', 'name_education_type', 'name_family_status', 'name_housing_type', 'occupation_type', 'cnt_children', 'cnt_fam_members']
numericals = ['amt_income_total', 'days_birth', 'days_employed']

#%%
cols = ['marital', 'gender', 'industry', 'ethnic']

#%%
def plot_dimension(col, cat_order):
    f, axes = plt.subplots(1, 2, figsize=(15, 7))
    # f.xticks(rotation = 45)
    sns.countplot(x = worldsUnpivot[(worldsUnpivot['dimension'] == col) & (worldsUnpivot['dataset'] == 'world1')]['value'], order = cat_order, ax = axes[0]).set_title("World1")
    sns.countplot(x = worldsUnpivot[(worldsUnpivot['dimension'] == col) & (worldsUnpivot['dataset'] == 'world2')]['value'], order = cat_order, ax = axes[1]).set_title("World2")
    axes[0].set(xlabel=col, ylabel='count')
    axes[1].set(xlabel=col, ylabel='count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='x', rotation=45)

#%%
# compare marital dimension between datasets
plot_dimension('marital', maritalDict.values())
print("Both populations have roughly the same distribution across marital status categories.")





















#%%
x, y, hue = "flag_own_car", "proportion", "code_gender"
hue_order = ["M", "F"]

(df[x]
.groupby(df[hue])
.value_counts(normalize=True)
.rename(y)
.reset_index()
.pipe((sns.barplot, "data"), x=x, y=y, hue=hue))

#%%
x, y, hue = "name_income_type", "proportion", "code_gender"
hue_order = ["M", "F"]

(df[x]
.groupby(df[hue])
.value_counts(normalize=True)
.rename(y)
.reset_index()
.pipe((sns.barplot, "data"), x=x, y=y, hue=hue))

#%%
def catbar(x, y, hue, hue_order):
    (df[x]
    .value_counts(normalize=True)
    .rename(y)
    .reset_index()
    .pipe((sns.barplot, "data"), x=x, y=y, hue=hue))

#%%
catbar("code_gender", "proportion", "name_income_type", ['M', 'F'])

#%%
df["name_income_type"]


#%%
df1 = sns.load_dataset("tips")
x, y, hue = "day", "proportion", "sex"
hue_order = ["Male", "Female"]

(df1[x]
 .groupby(df1[hue])
 .value_counts(normalize=True)
 .rename(y)
 .reset_index()
 .pipe((sns.barplot, "data"), x=x, y=y, hue=hue))
#%%
# print(df1[x])
df1.columns

#%%
df1


#%%





#%%
x, y, hue = "day", "proportion", "sex"
hue_order = ["Male", "Female"]

(df[x]
 .groupby(df[hue])
 .value_counts(normalize=True)
 .rename(y)
 .reset_index()
 .pipe((sns.barplot, "data"), x=x, y=y, hue=hue))










#%%
x, y, hue = "day", "proportion", "sex"
hue_order = ["Male", "Female"]

(df[x]
 .groupby(df[hue])
 .value_counts(normalize=True)
 .rename(y)
 .reset_index()
 .pipe((sns.barplot, "data"), x=x, y=y, hue=hue))










#%%
x,y = 'class', 'survived'

df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1)
g.ax.set_ylim(0,100)

for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
# %%
def countplt(feature):
    print(sns.countplot(y=feature, data=df, order = df[feature].value_counts().index))

#%%
countplt('CODE_GENDER')
#%%
ax = sns.countplot(y='OCCUPATION_TYPE', data=df, order = df['OCCUPATION_TYPE'].value_counts().index)

#%%
ax = sns.countplot(y='NAME_INCOME_TYPE', data=df, order = df['NAME_INCOME_TYPE'].value_counts().index)

#%%
sns.displot(df, x="AMT_INCOME_TOTAL", hue="NAME_INCOME_TYPE", kind="kde", multiple="stack")

#%%
sns.displot(df, x="AMT_INCOME_TOTAL", hue="NAME_INCOME_TYPE", kind="kde")

#%%

df2  = df[df['NAME_INCOME_TYPE']!= 'Student']
df2

#%%
# plot
# Initialize the FacetGrid object
pal = sns.cubehelix_palette(len(df2.NAME_INCOME_TYPE.unique()), rot=-.25, light=.7)
g = sns.FacetGrid(df2, row="NAME_INCOME_TYPE", hue="NAME_INCOME_TYPE", aspect=7, height=1, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "AMT_INCOME_TOTAL", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "AMT_INCOME_TOTAL", clip_on=False, color="w", lw=2, bw_adjust=.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
g.map(plt.ticklabel_format(style='plain', axis='x'))

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

g.map(label, "AMT_INCOME_TOTAL")

# g.plt.ticklabel_format(useOffset=False)

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

# uncomment the following line if there's a tight layout warning
g.fig.tight_layout()

#%%
















# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="NAME_INCOME_TYPE", hue="NAME_INCOME_TYPE", aspect=5, height=5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "AMT_INCOME_TOTAL",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "AMT_INCOME_TOTAL", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    # ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "AMT_INCOME_TOTAL")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

#%%









#%%
# create the data
x = df.AMT_INCOME_TOTAL.to_numpy()
g = df.NAME_INCOME_TYPE.to_numpy()
df2 = pd.DataFrame(dict(x=x, g=g))
# type(g)
x

#%%
# Create the data
# rs = np.random.RandomState(1979)
# x = rs.randn(500)
# g = np.tile(list("ABCDEFGHIJ"), 50)
# df2 = pd.DataFrame(dict(x=x, g=g))
# m = df2.g.map(ord)
# # df["x"] += m
# # type(m)
# print(df2)

#%%

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df2, row="g", hue="g", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
# g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "x")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)




#%%



# need to transform days birth, days employed
df.columns
# %%
df.DAYS_BIRTH
# %%
# vi test
# variance inflation chart