# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 00:15:48 2018

@author: Nsikan Udo
"""

import pandas as pd
import numpy as np #for mathematics calculations
import seaborn as sns #for data visualization
import matplotlib.pyplot as plt #for plotting graphs
#%matplotlib inline
#import warnings #To ignore any warnings
#warning.filterwarnings('ignore')

#Reading data into the program

train = pd.read_csv('C:\\Users\ALHASNA AGENCY\\Downloads\\train_u6lujuX_CVtuZ9i.csv')
test = pd.read_csv('C:\\Users\ALHASNA AGENCY\\Downloads\\test_Y3wMUE5_7gLdaTN.csv')

#Make a copy of the data set
train_original = train.copy()
teat_original = test.copy()

#Doing EDA for the data set

print(train.columns)
print(test.columns)

print(train.info)
print(train.describe())

print(train.dtypes)
print(test.shape)
print(len(train) + len(test))

""" We look at the target variable. Since it is a categorical variable, 
We examine it frequency table, percentage distribution and bar plot """

my_target = train['Loan_Status'].value_counts()
print(my_target)

# Normalize can be set to True to print proportions instead of number 
my_target_prop = train['Loan_Status'].value_counts(normalize=True)
print(my_target_prop)

#Plot bar plot of Loan_Status
train['Loan_Status'].value_counts().plot.bar(title='Loan Status')
plt.show()

#Plot bar for independent categorical variables

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize = (20,10), title = 'Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

plt.show()

#Plot bar for independent variable (Ordinal)

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

plt.show()

#Plot Independent variables (Numeric)

plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize = (16,5))
plt.show()

#Segregate the ApplicantIncome by Education

train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")

#Ploting Categorical Independent Variables vs Target Variabe

Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()


"""Numerical Independent Variable vs Target Variable
We will try to find the mean income of people for which the loan has been approved 
vs the mean income of people for which the loan has not been approved."""

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()







