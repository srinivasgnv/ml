#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:13:40 2019

@author: Srinivas Gannavarapu
"""

import pandas as pd
import sklearn as sk
import math
import numpy as np
from scipy import stats
import matplotlib as matlab
import statsmodels

###############LAB:Correlation Calculation########################

#Dataset: Air Travel Data\Air_travel.csv
#Importing Air passengers data
air = pd.read_csv("./Data/AirPassengers.csv")

# describing/inspecting data
print (air.shape)
print(air.columns.values)
print (air.head(10))
print (air.describe())

#Find the correlation between number of passengers and promotional budget.
np.corrcoef(air.Passengers,air.Promotion_Budget)

#Find the correlation between number of passengers and Intere metro flight ration.
np.corrcoef(air.Passengers,air.Inter_metro_flight_ratio)

#Find the correlation between number of passengers and Service Quality Score .
np.corrcoef(air.Passengers,air.Service_Quality_Score)

#Draw a scatter plot between number of passengers and promotional budget
matlab.pyplot.scatter(air.Passengers, air.Promotion_Budget)

#Find the correlation between number of passengers and Service_Quality_Score
np.corrcoef(air.Passengers,air.Service_Quality_Score)


##############################################Regression######################################

#Correlation between promotion and passengers count
np.corrcoef(air.Passengers,air.Promotion_Budget)

#Draw a scatter plot between   Promotion_Budget and Passengers. Is there any any pattern between Promotion_Budget and Passengers?
matlab.pyplot.scatter(air.Promotion_Budget,air.Passengers)

#Build a linear regression model and estimate the expected passengers for a Promotion_Budget is 650,000
##Regression Model  promotion and passengers count
import statsmodels.formula.api as sm
model = sm.ols(formula='Passengers ~ Promotion_Budget', data=air)
model
fitted1 = model.fit()
fitted1.summary()

# Passengers = 1259.6 + 0.0695 * PromotionBudget

print(1259.6 + (0.0695)*700000)


# Building another model for inter_metor_flight_ratio 
import statsmodels.formula.api as sm2
model2 = sm2.ols(formula='Passengers ~ Inter_metro_flight_ratio', data=air)
model2
fitted2 = model2.fit()
fitted2.summary()

# Building multiple regression model 
import statsmodels.formula.api as sm3
model3 = sm3.ols(formula='Passengers ~  Service_Quality_Score', data=air)
model3
fitted3 = model3.fit()
fitted3.summary()


#Building the same model using sci-kit learn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(air[["Promotion_Budget"]], air[["Passengers"]])

#Coefficients
lr.coef_
lr.intercept_

#Build a regression line to predict the passengers using Inter_metro_flight_ratio

##Regression Model inter_metro_flight_ratio and passengers count
matlab.pyplot.scatter(air.Inter_metro_flight_ratio,air.Passengers)

import statsmodels.formula.api as sm
model = sm.ols(formula='Passengers ~ Inter_metro_flight_ratio', data=air)
fitted2 = model.fit()
fitted2.summary()

#Building the same model using sci-kit learn
#from sklearn.linear_model import LinearRegression
#lr = LinearRegression()
#lr.fit(air[["Inter_metro_flight_ratio"]], air[["Passengers"]])

#Coefficients
#lr.coef_
#lr.intercept_


#############################################################################
############ Lab:R Sqaure ##################
#What is the R-square value of Passengers vs Promotion_Budget model?
fitted1.summary()

#What is the R-square value of Passengers vs Inter_metro_flight_ratio

fitted2.summary()


################################################
#############Lab: Multiple Regerssion Model ####################
#Build a multiple regression model to predict the number of passengers

import statsmodels.formula.api as sm
model = sm.ols(formula='Passengers ~ Promotion_Budget+Service_Quality_Score+Inter_metro_flight_ratio', data=air)
fitted = model.fit()
fitted.summary()

#What is R-square value
fitted.summary()

#Are there any predictor variables that are not impacting the dependent variable 
##Inter_metro_flight_ratio is dropped
import statsmodels.formula.api as sm
model = sm.ols(formula='Passengers ~ Promotion_Budget+Service_Quality_Score', data=air)
fitted = model.fit()
fitted.summary()
 

###############################################
##Adjusted R-Square

adj_sample=pd.read_csv("/Users/ibm/Downloads/Adj_Sample.csv")
#Build a model to predict y using x1,x2 and x3. Note down R-Square and Adj R-Square values 
model = sm.ols(formula='Y ~ x1+x2+x3', data=adj_sample)
fitted = model.fit()
fitted.summary()
#R-Squared 

#Model2
model = sm.ols(formula='Y ~ x1+x2+x3+x4+x5+x6', data=adj_sample)
fitted = model.fit()
fitted.summary()

#Model3
model = sm.ols(formula='Y ~ x1+x2+x3+x4+x5+x6+x7+x8', data=adj_sample)
fitted = model.fit()
fitted.summary()

#################################################################################3
#####Multiple Regression- issues
    
#Import Final Exam Score data
final_exam=pd.read_csv("/Users/ibm/Downloads/Final Exam Score.csv")

#Size of the data
print(final_exam.shape)

#Variable names
print(final_exam.columns)

#First few observations
print(final_exam.head(10))

#Build a model to predict final score using the rest of the variables.

import statsmodels.formula.api as sm_exam
model_exam = sm_exam.ols(formula='Final_exam_marks ~ Sem1_Science+Sem2_Science+Sem2_Math', data=final_exam)
fitted_exam = model_exam.fit()
print(fitted_exam.summary())
print(fitted_exam.rsquared)


#How are Sem2_Math & Final score related? As Sem2_Math score increases, what happens to Final score? 

#Remove "Sem1_Math" variable from the model and rebuild the model
import statsmodels.formula.api as sm
model2 = sm.ols(formula='Final_exam_marks ~ Sem1_Science+Sem2_Science+Sem2_Math', data=final_exam)
fitted2 = model2.fit()
fitted2.summary()


#Is there any change in R square or Adj R square

#How are Sem2_Math  & Final score related now? As Sem2_Math score increases, what happens to Final score? 

#Scatter Plot between the predictor variables
matlab.pyplot.scatter(final_exam.Sem1_Math,final_exam.Sem2_Math)

#Find the correlation between Sem1_Math & Sem2_Math 
np.correlate(final_exam.Sem1_Math,final_exam.Sem2_Math)

########################Multicollinearity detection#########################
##Testing Multicollinearity

model1 = sm.ols(formula='Final_exam_marks ~ Sem1_Science+Sem2_Science+Sem1_Math+Sem2_Math', data=final_exam)
fitted1 = model1.fit()
fitted1.summary()
fitted1.summary2()

#Code for VIF Calculation

#Writing a function to calculate the VIF values

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
vif_cal(input_data=final_exam, dependent_col="Final_exam_marks")

#VIF Values given by statsmodels.stats.outliers_influence.variance_inflation_factor are not accurate
#import statsmodels.stats.outliers_influence
#help(statsmodels.stats.outliers_influence.variance_inflation_factor)
#statsmodels.stats.outliers_influence.variance_inflation_factor(final_exam.drop(["Final_exam_marks"], axis=1).as_matrix(), 0)
#statsmodels.stats.outliers_influence.variance_inflation_factor(final_exam.drop(["Final_exam_marks"], axis=1).as_matrix(), 1)
#statsmodels.stats.outliers_influence.variance_inflation_factor(final_exam.drop(["Final_exam_marks"], axis=1).as_matrix(), 2)
#statsmodels.stats.outliers_influence.variance_inflation_factor(final_exam.drop(["Final_exam_marks"], axis=1).as_matrix(), 3)


import statsmodels.formula.api as sm
model2 = sm.ols(formula='Final_exam_marks ~ Sem1_Science+Sem2_Science+Sem2_Math', data=final_exam)
fitted2 = model2.fit()
fitted2.summary()

vif_cal(input_data=final_exam.drop(["Sem1_Math"], axis=1), dependent_col="Final_exam_marks")
vif_cal(input_data=final_exam.drop(["Sem1_Math","Sem1_Science"], axis=1), dependent_col="Final_exam_marks")

## VIF


###############################################
##Multiple Regression model building 
print("Loading webpage product sales and describing the data")
Webpage_Product_Sales=pd.read_csv("/Users/ibm/Downloads/Webpage_Product_Sales.csv")
print("Shape --> ",Webpage_Product_Sales.shape)
print("Columns --> ",Webpage_Product_Sales.columns)

print("Build the model with most of the variables to predict the target i.e sales")
import statsmodels.formula.api as sm_webpage
model_webpage = sm_webpage.ols(formula='Sales ~ Web_UI_Score+Server_Down_time_Sec+Holiday+Special_Discount+Clicks_From_Serach_Engine+Online_Ad_Paid_ref_links+Social_Network_Ref_links+Month+Weekday+DayofMonth', data=Webpage_Product_Sales)
fitted_webpage = model_webpage.fit()
fitted_webpage.summary()

print("ReBuild the model with optimal variables to predict the target i.e sales. Dropping Web_UI_Score & Clicks_From_Serach_Engine as it's P value is > 0.05 ")
import statsmodels.formula.api as sm_webpage
model_webpage = sm_webpage.ols(formula='Sales ~ Server_Down_time_Sec+Holiday+Special_Discount+Online_Ad_Paid_ref_links+Social_Network_Ref_links+Month+Weekday+DayofMonth', data=Webpage_Product_Sales)
fitted_webpage = model_webpage.fit()
fitted_webpage.summary()

#VIF
print ("Initial Calculate VIF ")
vif_cal(Webpage_Product_Sales,"Sales")

#
#Web_UI_Score+Server_Down_time_Sec+Holiday+Special_Discount+Clicks_From_Serach_Engine+Online_Ad_Paid_ref_links+Social_Network_Ref_links+Month+Weekday+DayofMonth
print ("We found there are two VIF values >= 5 so they have to be dropped from the model and reclculate VIF")
##Dropped Clicks_From_Serach_Engine and Online_Ad_Paid_ref_links based on VIF
import statsmodels.formula.api as sm
model2 = sm.ols(formula='Sales ~ Web_UI_Score+Server_Down_time_Sec+Holiday+Special_Discount+Online_Ad_Paid_ref_links+Social_Network_Ref_links+Month+Weekday+DayofMonth', data=Webpage_Product_Sales)
fitted2 = model2.fit()
fitted2.summary()

#VIF for the updated model
#vif_cal(input_data=final_exam.drop(["Sem1_Math","Sem1_Science"], axis=1), dependent_col="Final_exam_marks")
vif_cal(Webpage_Product_Sales.drop(["Clicks_From_Serach_Engine","Web_UI_Score"],axis=1),"Sales")
#No VIF is more than 5


##Drop the less impacting variables based on p-values.
##Dropped Web_UI_Score based on P-value

print("Final model")
import statsmodels.formula.api as sm
model3 = sm.ols(formula='Sales ~ Server_Down_time_Sec+Holiday+Special_Discount+Online_Ad_Paid_ref_links+Social_Network_Ref_links+Month+Weekday+DayofMonth', data=Webpage_Product_Sales)
fitted3 = model3.fit()
fitted3.summary()


#How many variables are there in the final model?
#10
#What is the R-squared of the final model?
print("Number of vars in final model ",8)
print("R Squared for final model is ",81)

# There are two ways of eleminating the vars (1) P value (2) VIF which one to use first. I see difference 
# How to use the model to predict sales 
# 

























