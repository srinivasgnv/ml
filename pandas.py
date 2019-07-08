#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:54:27 2019

@author: ibm
"""

import pandas as pd
income= pd.read_csv ('./data/Income_data.csv')
 
#print (income.columns)
#print (income.dtypes)
#print (income["capital-gain"].mean())
#print (income["capital-gain"].median())

#usa_income = income[income["native-country"]==' United-States']
#print(usa_income.shape)

#other_income = income[income["native-country"]!=' United-States']
#print (other_income.shape)

#USA
#var_usa = usa_income["education-num"].var()
#print(var_usa)

#Other
#var_other = other_income["education-num"].var()
#print(var_other)

#std_usa = usa_income["education-num"].std()
#print(std_usa)

#std_other = other_income["education-num"].std()
#print(std_other)



#print(income['capital-gain'].quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]))

print(income['hours-per-week'].quantile([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]))