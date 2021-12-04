#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
#%%
application = pd.read_csv('/Users/chim/Downloads/application_record.csv')
credit = pd.read_csv('/Users/chim/Downloads/credit_record.csv')
#%%
application.describe()
credit.describe()

#%%
application.info()
credit.info()
# %%
application.isnull().sum()
#%%
application.drop('OCCUPATION_TYPE', axis=1, inplace=True)
# alot of null in this column we i think it's best to take this out
# %%
# Droppin duplicates if there's any buy leaving the last entry of the duplicate leaving one 
len(application['ID']) - len(application['ID'].unique())
application = application.drop_duplicates('ID', keep='last') 

#%%
# Changing the days birth to age 
application['DAYS_BIRTH'] = application['DAYS_BIRTH'].map(lambda x: round(x / -365))
application.rename(columns = {'DAYS_BIRTH': 'AGE'}, inplace=True)
# %%
# if 'DAYS_EMPLOYED' is positive no, it means individual is currently unemployed, so lets replace it with 0
application.loc[application['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 0
#%%
# Changind DAYS_UNEMPLOYED to years and YEARS_EMPLOYED
application['DAYS_EMPLOYED'] = abs(round(application['DAYS_EMPLOYED']/-365,0))
application.rename(columns={'DAYS_EMPLOYED':'YEARS_EMPLOYED'}, inplace=True)
# %%
# Now let's see if we need all those columns

 # 1   CODE_GENDER          438510 non-null  object 
 # 2   FLAG_OWN_CAR         438510 non-null  object 
 # 3   FLAG_OWN_REALTY      438510 non-null  object 
 # 6   NAME_INCOME_TYPE     438510 non-null  object 
 # 7   NAME_EDUCATION_TYPE  438510 non-null  object 
 # 8   NAME_FAMILY_STATUS   438510 non-null  object 
 # 9   NAME_HOUSING_TYPE    438510 non-null  object 



# 4   CNT_CHILDREN         438510 non-null  int64  
# 5   AMT_INCOME_TOTAL     438510 non-null  float64
# 10  AGE                  438510 non-null  int64  
# 11  YEARS_EMPLOYED       438510 non-null  float64
# 12  FLAG_MOBIL           438510 non-null  int64  
# 13  FLAG_WORK_PHONE      438510 non-null  int64  
# 14  FLAG_PHONE           438510 non-null  int64  
# 15  FLAG_EMAIL           438510 non-null  int64  
# 16  CNT_FAM_MEMBERS      438510 non-null  float64

#%%
# Probably don't need this columns, just tells us whether they have a phone or email
application.drop('FLAG_MOBIL', axis=1, inplace=True)
application.drop('FLAG_WORK_PHONE', axis=1, inplace=True)
application.drop('FLAG_EMAIL', axis=1, inplace=True)

# %%
application
# %%
# Checking for outliers in the numerical columns left: Run one by one
sns.boxplot(application['CNT_CHILDREN']) # has outliers 
sns.boxplot(application['AMT_INCOME_TOTAL']) # has outliers
sns.boxplot(application['AGE']) # checks out
sns.boxplot(application['YEARS_EMPLOYED']) # has outliers 
sns.boxplot(application['CNT_FAM_MEMBERS']) # has outliers

#%%
# REviewed documentaion on this, to remove outliers
high_bound = application['CNT_CHILDREN'].quantile(0.999)
print('high_bound :', high_bound)
low_bound = application['CNT_CHILDREN'].quantile(0.001)
print('low_bound :', low_bound)

#
application = application[(application['CNT_CHILDREN']>=low_bound) & (application['CNT_CHILDREN']<=high_bound)]

#%%
high_bound = application['AMT_INCOME_TOTAL'].quantile(0.999)
print('high_bound :', high_bound)
low_bound = application['AMT_INCOME_TOTAL'].quantile(0.001)
print('low_bound :', low_bound)

#
application = application[(application['AMT_INCOME_TOTAL']>=low_bound) & (application['AMT_INCOME_TOTAL']<=high_bound)]

#%%
high_bound = application['YEARS_EMPLOYED'].quantile(0.999)
print('high_bound :', high_bound)
low_bound = application['YEARS_EMPLOYED'].quantile(0.001)
print('low_bound :', low_bound)

#
application = application[(application['YEARS_EMPLOYED']>=low_bound) & (application['YEARS_EMPLOYED']<=high_bound)]


#%%
high_bound = application['CNT_FAM_MEMBERS'].quantile(0.999)
print('high_bound :', high_bound)
low_bound = application['CNT_FAM_MEMBERS'].quantile(0.001)
print('low_bound :', low_bound)

application = application[(application['CNT_FAM_MEMBERS']>=low_bound) & (application['CNT_FAM_MEMBERS']<=high_bound)]

# %%
application
# everything should be good now let's clean up credit

#%%
credit.info()
credit.describe()

#%%
credit.isnull().sum()
# no nulls
# %%
credit['STATUS'].value_counts()

# %%
# 0 if they have any balance of any kind
# 1 if they don't have a balance 
credit['STATUS'].replace(['C', 'X'],0, inplace=True)
credit['STATUS'].replace(['2','3','4','5'],1, inplace=True)

#%%
# make it an integer
credit['STATUS'] = credit['STATUS'].astype('int')

# %%
credit['STATUS'].value_counts()
# there are too many 0s in the column
#%%
# grouping it by ID so we can add it to 
creditgroup = credit.groupby('ID').agg(max).reset_index()
# %%
# joining
appcre = pd.merge(application, creditgroup, on='ID', how='inner')

# %%
appcre.head()
# %%
#plt.figure(figsize = (8,8))
import matplotlib.pyplot as plt

plt.figure(figsize = (8,8))
sns.heatmap(appcre.corr(), annot=True)
plt.show()

#nothin is correlated with status
# %%
