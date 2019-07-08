#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:54:57 2019

@author: ibm
"""

##############################################################################
####################### logistic Regresssion ###################################

##### LAB-What is the need of logistic regression ################

#Import the Dataset: Product Sales Data/Product_sales.csv
import numpy as np
import pandas as pd
import matplotlib as plt
import scipy as sp
sales=pd.read_csv("./data/Product_sales.csv")

#What are the variables in the dataset? 
print(sales.columns.values)

#Build a predictive model for Bought vs Age

### we need to use the statsmodels package, which enables many statistical methods to be used in Python
#import statsmodels.formula.api as sm
#from statsmodels.formula.api import ols
#model = sm.ols(formula='Bought ~ Age', data=sales)
#fitted = model.fit()
#fitted.summary()

#If Age is 4 then will that customer buy the product?

import sklearn as sk
from sklearn import linear_model

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(sales[["Age"]], sales[["Bought"]])

print(lr.coef_)
print(lr.intercept_)

d1=pd.DataFrame({"age1":[4]})
predict1=lr.predict(d1)

predict1=lr.predict(d1)
print(predict1)

### for age=4 value is less than zero  

#If Age is 105 then will that customer buy the product?
#age2=105

d2=pd.DataFrame({"age2" :[105]})
predict2=lr.predict(d2)
print(predict2)
##### for age=105 value is greater than one.

#######From this linear regression,we can not interpret whether a person buys or not

################ Lab: Logistic Regression ######################

#Dataset: Product Sales Data/Product_sales.csv
sales=pd.read_csv("./data/Product_sales.csv")


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(sales[["Age"]],sales["Bought"])

print(logistic.coef_)
print(logistic.intercept_)
#A 4 years old customer, will he buy the product?
d1=pd.DataFrame({"age1":[4]})

predict_age1=logistic.predict(d1)
print(predict_age1)

#If Age is 105 then will that customer buy the product?
d1=pd.DataFrame({"age1":[105]})
predict_age2=logistic.predict(d1)
print(predict_age2)

##############LAB: Multiple Logistic Regression####################

#Dataset: Fiberbits/Fiberbits.csv
Fiber=pd.read_csv("./data/Fiberbits.csv")
list(Fiber.columns.values)  ###to get variables list

#Build a model to predict the chance of attrition for a given customer using all the features. 
from sklearn.linear_model import LogisticRegression
logistic1= LogisticRegression()
###fitting logistic regression for active customer on rest of the variables#######
logistic1.fit(Fiber[["income"]+['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['monthly_bill']+['technical_issues_per_month']+['Speed_test_result']],Fiber[['active_cust']])

print(logistic1.coef_)
print(logistic1.intercept_)

# check if the model is accurate - using confustion matrix 


###############LAB: Confusion Matrix & Accuracy ########################
### calculate confusion matrix
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix###for using confusion matrix###

predict1=logistic1.predict(Fiber[["income"]+['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['monthly_bill']+['technical_issues_per_month']+['Speed_test_result']])
predict1

cm1 = confusion_matrix(Fiber[['active_cust']],predict1)
print(cm1)

#####from confusion matrix calculate accuracy
total1=sum(sum(cm1))
print(total1)

accuracy1=(cm1[0,0]+cm1[1,1])/total1
accuracy1

###########  LAB-Multicollinearity################3
#Is there any multicollinearity in fiber bits model? 
#Identify and remove multicollinearity from the model

def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

#Calculating VIF values using that function
vif_cal(input_data=Fiber, dependent_col="active_cust")


###################LAB: Individual Impact of Variables################

#Identify top impacting and least impacting variables in fiber bits models
####Variable importance is decided from Wald chi‐square value i.e square of Z parameter.
##from summary,sort the varibles in the order of the squares of their z values.
#Find the variable importance and order them based on their impact

import statsmodels.formula.api as sm

m1=sm.Logit(Fiber['active_cust'],Fiber[["income"]+['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['monthly_bill']+['technical_issues_per_month']+['Speed_test_result']])
m1
m1.fit()
#m1.fit().summary()
m1.fit().summary2()
