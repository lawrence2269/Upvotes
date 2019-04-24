#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:20:01 2019

@author: lawrence
"""

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import skew

#Loading the data
train = pd.read_csv("/Users/lawrence/Documents/Capstone Project-1/train_upvotes.csv")
train.shape

#Calculating correlation coefficient
np.corrcoef(train['Reputation'],train['Views']) #0.03
np.corrcoef(train['Answers'],train['Views']) #0.49
np.corrcoef(train['Answers'],train['Reputation']) #0.06
np.corrcoef(train['Username'],train['Reputation'])#-0.04
np.corrcoef(train['Username'],train['Views']) #0.002
np.corrcoef(train['Reputation'],train['Upvotes']) #0.26
np.corrcoef(train['Answers'],train['Upvotes']) #0.20
np.corrcoef(train['Views'],train['Upvotes']) #0.43
np.corrcoef(train['Username'],train['Upvotes']) #-0.011

#Removing outliers
sns.boxplot(train['Reputation'])
sns.boxplot(train['Answers'])
sns.boxplot(train['Views'])

#Training Data- Reputation outlier removal
train_Reputation_Q1 = train['Reputation'].quantile(0.25)
train_Reputation_Q3 = train['Reputation'].quantile(0.75)
train_Reputation_IQR = train_Reputation_Q3 - train_Reputation_Q1
print(train_Reputation_Q1,train_Reputation_Q3,train_Reputation_IQR)
train = train[~((train.Reputation<(train_Reputation_Q1-1.5*train_Reputation_IQR))|(train.Reputation>(train_Reputation_Q3+1.5*train_Reputation_IQR)))]
sns.boxplot(train['Reputation'])

#Training Data- Answers outlier removal
train_Answers_Q1 = train['Answers'].quantile(0.25)
train_Answers_Q3 = train['Answers'].quantile(0.75)
train_Answers_IQR = train_Answers_Q3 - train_Answers_Q1
print(train_Answers_Q1,train_Answers_Q3,train_Answers_IQR)
train = train[~((train.Answers<(train_Answers_Q1-1.5*train_Answers_IQR))|(train.Answers>(train_Answers_Q3+1.5*train_Answers_IQR)))]
sns.boxplot(train['Answers'])

#Training Data- Answers outlier removal
train_Views_Q1 = train['Views'].quantile(0.25)
train_Views_Q3 = train['Views'].quantile(0.75)
train_Views_IQR = train_Views_Q3 - train_Views_Q1
print(train_Views_Q1,train_Views_Q3,train_Views_IQR)
train = train[~((train.Views<(train_Views_Q1-1.5*train_Views_IQR))|(train.Views>(train_Views_Q3+1.5*train_Views_IQR)))]
sns.boxplot(train['Views'])

#Missing value imputation
train.isna().sum()

#Normalizing the data
sns.distplot(train['Reputation'])
print(round(skew(train['Reputation']),2)) #1.14

sns.distplot(train['Answers'])
print(round(skew(train['Answers']),2)) #0.94

sns.distplot(train['Views'])
print(round(skew(train['Views']),2)) #0.99

print("Before Cbrt Transformation:",skew(train['Reputation']))
print("After Cbrt Transformation:",skew(np.cbrt(train['Reputation'])))
sns.distplot(np.cbrt(train['Reputation']))

print("Before Sqrt Transformation:",skew(train['Answers']))
print("After Sqrt Transformation:",skew(np.sqrt(train['Answers'])))
sns.distplot(np.sqrt(train['Answers']))

print("Before Cbrt Transformation:",skew(train['Views']))
print("After Cbrt Transformation:",skew(np.cbrt(train['Views'])))
sns.distplot(np.cbrt(train['Views']))

train['Reputation'] = np.cbrt(train['Reputation'])
train['Views'] = np.cbrt(train['Views'])
train['Answers'] = np.sqrt(train['Answers'])

#One hot encoding
train = pd.get_dummies(train)

y = train['Upvotes']
x = train.drop(['Upvotes','ID','Username'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x)

"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
#regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
round(r2_score(y_test,y_pred),2) #0.05

from sklearn.metrics import mean_squared_error
print(round(np.sqrt(mean_squared_error(y_test,y_pred)),2)) #44.77

#############################Preprocessing Test Data###############################
test = pd.read_csv("/Users/lawrence/Documents/Capstone Project-1/test_upvotes.csv")

np.set_printoptions(suppress=True)

#Calculating correlation coefficient
np.corrcoef(test['Reputation'],test['Views']) #0.03
np.corrcoef(test['Answers'],test['Views']) #0.51
np.corrcoef(test['Answers'],test['Reputation']) #0.07

#Removing outliers
sns.boxplot(test['Reputation'])
sns.boxplot(test['Answers'])
sns.boxplot(test['Views'])

#Training Data- Reputation outlier removal
test_Reputation_Q1 = test['Reputation'].quantile(0.25)
test_Reputation_Q3 = test['Reputation'].quantile(0.75)
test_Reputation_IQR = test_Reputation_Q3 - test_Reputation_Q1
print(test_Reputation_Q1,test_Reputation_Q3,test_Reputation_IQR)
test = test[~((test.Reputation<(test_Reputation_Q1-1.5*test_Reputation_IQR))|(test.Reputation>(test_Reputation_Q3+1.5*test_Reputation_IQR)))]
sns.boxplot(test['Reputation'])

#Training Data- Answers outlier removal
test_Answers_Q1 = test['Answers'].quantile(0.25)
test_Answers_Q3 = test['Answers'].quantile(0.75)
test_Answers_IQR = test_Answers_Q3 - test_Answers_Q1
print(test_Answers_Q1,test_Answers_Q3,test_Answers_IQR)
test = test[~((test.Answers<(test_Answers_Q1-1.5*test_Answers_IQR))|(test.Answers>(test_Answers_Q3+1.5*test_Answers_IQR)))]
sns.boxplot(test['Answers'])

#Training Data- Answers outlier removal
test_Views_Q1 = test['Views'].quantile(0.25)
test_Views_Q3 = test['Views'].quantile(0.75)
test_Views_IQR = test_Views_Q3 - test_Views_Q1
print(test_Views_Q1,test_Views_Q3,test_Views_IQR)
test = test[~((test.Views<(test_Views_Q1-1.5*test_Views_IQR))|(test.Views>(test_Views_Q3+1.5*test_Views_IQR)))]
sns.boxplot(test['Views'])

#Missing value imputation
test.isna().sum()

#Normalizing the data
sns.distplot(test['Reputation'])
print(round(skew(test['Reputation']),2)) #1.14

sns.distplot(test['Answers'])
print(round(skew(test['Answers']),2)) #0.94

sns.distplot(test['Views'])
print(round(skew(test['Views']),2)) #0.99

print("Before Cbrt Transformation:",skew(test['Reputation']))
print("After Cbrt Transformation:",skew(np.cbrt(test['Reputation'])))
sns.distplot(np.cbrt(test['Reputation']))

print("Before Sqrt Transformation:",skew(test['Answers']))
print("After Sqrt Transformation:",skew(np.sqrt(test['Answers'])))
sns.distplot(np.sqrt(test['Answers']))

print("Before Cbrt Transformation:",skew(test['Views']))
print("After Cbrt Transformation:",skew(np.cbrt(test['Views'])))
sns.distplot(np.cbrt(test['Views']))

test['Reputation'] = np.cbrt(test['Reputation'])
test['Views'] = np.cbrt(test['Views'])
test['Answers'] = np.sqrt(test['Answers'])

#One hot encoding
test = pd.get_dummies(test)

Id = pd.DataFrame(test['ID'],columns=['ID'])
test = test.drop(['ID','Username'],axis=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test = sc.fit_transform(test)

#Predicting the values
y_test_pred = regressor.predict(test)

#Writing the y_test_pred to csv
upVotes_Predicted = pd.DataFrame(y_test_pred,columns=["Upvotes"])

Id.reset_index(drop = True,inplace=True)
upVotes_Predicted.reset_index(drop = True,inplace=True)

finalData = pd.concat([Id,upVotes_Predicted],axis=1)

#To CSV
finalData.to_csv("/Users/lawrence/Documents/Capstone Project-1/Predicted_Upvotes_Test_Data.csv",index = False)


