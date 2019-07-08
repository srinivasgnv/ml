#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:45:57 2019

@author: ibm
"""

###########################################################################
###############Neural Network Code

####################LAB: Logistic Regression######################
###Data
import pandas as pd
Emp_Purchase_raw = pd.read_csv("./data/Emp_Purchase.csv")
Emp_Purchase_raw.shape
Emp_Purchase_raw.columns.values
Emp_Purchase_raw.head(10)

####Filter the data and take a subset from above dataset . Filter condition is Sample_Set<3
Emp_Purchase1=Emp_Purchase_raw[Emp_Purchase_raw.Sample_Set<3]
Emp_Purchase1.shape
Emp_Purchase1.columns.values
Emp_Purchase1.head(10)

#frequency table of Purchase variable
Emp_Purchase1.Purchase.value_counts()

####The clasification graph
#Draw a scatter plot that shows Age on X axis and Experience on Y-axis. Try to distinguish the two classes with colors or shapes.
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(Emp_Purchase1.Age[Emp_Purchase1.Purchase==0],Emp_Purchase1.Experience[Emp_Purchase1.Purchase==0], s=15, c='b', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase1.Age[Emp_Purchase1.Purchase==1],Emp_Purchase1.Experience[Emp_Purchase1.Purchase==1], s=15, c='r', marker="+", label='Purchase 1')

plt.xlim(min(Emp_Purchase1.Age), max(Emp_Purchase1.Age))
plt.ylim(min(Emp_Purchase1.Experience), max(Emp_Purchase1.Experience))
plt.legend(loc='upper left');

plt.show()

###Logistic Regerssion model1
import statsmodels.formula.api as sm
model1 = sm.logit(formula='Purchase ~ Age+Experience', data=Emp_Purchase1)
fitted1 = model1.fit()
fitted1.summary2()

#######Accuracy and error of the model1
#Create the confusion matrix
predicted_values=fitted1.predict(Emp_Purchase1[["Age"]+["Experience"]])
predicted_values[1:10]
threshold=0.5

import numpy as np
predicted_class=np.zeros(predicted_values.shape)
predicted_class[predicted_values>threshold]=1

predicted_class

from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Emp_Purchase1[['Purchase']],predicted_class)
print(ConfusionMatrix)
accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print('Accuracy : ',accuracy)
error=1-accuracy
print('Error: ',error)


#coefficients
slope1=fitted1.params[1]/(-fitted1.params[2])
intercept1=fitted1.params[0]/(-fitted1.params[2])


#Finally draw the decision boundary for this logistic regression model
      
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(Emp_Purchase1.Age[Emp_Purchase1.Purchase==0],Emp_Purchase1.Experience[Emp_Purchase1.Purchase==0], s=10, c='b', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase1.Age[Emp_Purchase1.Purchase==1],Emp_Purchase1.Experience[Emp_Purchase1.Purchase==1], s=10, c='r', marker="+", label='Purchase 1')

plt.xlim(min(Emp_Purchase1.Age), max(Emp_Purchase1.Age))
plt.ylim(min(Emp_Purchase1.Experience), max(Emp_Purchase1.Experience))
plt.legend(loc='upper left');

x_min, x_max = ax1.get_xlim()
ax1.plot([0, x_max], [intercept1, x_max*slope1+intercept1])
plt.show()

############################################
####Overall Data LAB: Non-Linear Decision Boundaries
############################################

#plotting the overall data
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==0], s=10, c='b', marker="o", label='Purchase 0')
ax.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==1], s=10, c='r', marker="+", label='Purchase 1')

plt.xlim(min(Emp_Purchase_raw.Age), max(Emp_Purchase_raw.Age))
plt.ylim(min(Emp_Purchase_raw.Experience), max(Emp_Purchase_raw.Experience))
plt.legend(loc='upper left');
plt.show()

###Logistic Regerssion model1
import statsmodels.formula.api as sm
model = sm.logit(formula='Purchase ~ Age+Experience', data=Emp_Purchase_raw)
fitted = model.fit()
fitted.summary2()

# getting slope and intercept of the line
slope=fitted.params[1]/(-fitted.params[2])
intercept=fitted.params[0]/(-fitted.params[2])

#Finally draw the decision boundary for this logistic regression model
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==0], s=10, c='b', marker="o", label='Purchase 0')
ax.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==1], s=10, c='r', marker="+", label='Purchase 1')
plt.xlim(min(Emp_Purchase_raw.Age), max(Emp_Purchase_raw.Age))
plt.ylim(min(Emp_Purchase_raw.Experience), max(Emp_Purchase_raw.Experience))
plt.legend(loc='upper left');

x_min, x_max = ax.get_xlim()
ax.plot([0, x_max], [intercept, x_max*slope+intercept])
plt.show()

#######Accuracy and error of the model1
#Create the confusion matrix
#predicting values
predicted_values=fitted.predict(Emp_Purchase_raw[["Age"]+["Experience"]])
predicted_values[1:10]

#Lets convert them to classes using a threshold
threshold=0.5
threshold

import numpy as np
predicted_class=np.zeros(predicted_values.shape)
predicted_class[predicted_values>threshold]=1

#Predcited Classes
predicted_class[1:10]

from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Emp_Purchase_raw[['Purchase']],predicted_class)
print(ConfusionMatrix)
accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print(accuracy)

error=1-accuracy
error

############################################
#### sample-2   
############################################

Emp_Purchase2=Emp_Purchase_raw[Emp_Purchase_raw.Sample_Set>1]
Emp_Purchase2.shape
Emp_Purchase2.columns.values
Emp_Purchase2.head(10)

#frequency table of Purchase variable
Emp_Purchase2.Purchase.value_counts()

####The clasification graph
#Draw a scatter plot that shows Age on X axis and Experience on Y-axis. Try to distinguish the two classes with colors or shapes.
import matplotlib.pyplot as plt

fig = plt.figure()
ax2 = fig.add_subplot(111)

ax2.scatter(Emp_Purchase2.Age[Emp_Purchase2.Purchase==0],Emp_Purchase2.Experience[Emp_Purchase2.Purchase==0], s=10, c='b', marker="o", label='Purchase 0')
ax2.scatter(Emp_Purchase2.Age[Emp_Purchase2.Purchase==1],Emp_Purchase2.Experience[Emp_Purchase2.Purchase==1], s=10, c='r', marker="+", label='Purchase 1')
plt.xlim(min(Emp_Purchase2.Age), max(Emp_Purchase2.Age))
plt.ylim(min(Emp_Purchase2.Experience), max(Emp_Purchase2.Experience))
plt.legend(loc='upper left');
plt.show()

###Logistic Regerssion model1
import statsmodels.formula.api as sm
model2 = sm.logit(formula='Purchase ~ Age+Experience', data=Emp_Purchase2)
fitted2 = model2.fit(method="bfgs")
fitted2.summary2()

# getting slope and intercept of the line
# getting slope and intercept of the line
slope2=fitted2.params[1]/(-fitted2.params[2])
intercept2=fitted2.params[0]/(-fitted2.params[2])

#Finally draw the decision boundary for this logistic regression model
import matplotlib.pyplot as plt

fig = plt.figure()
ax2 = fig.add_subplot(111)

ax2.scatter(Emp_Purchase2.Age[Emp_Purchase2.Purchase==0],Emp_Purchase2.Experience[Emp_Purchase2.Purchase==0], s=10, c='b', marker="o", label='Purchase 0')
ax2.scatter(Emp_Purchase2.Age[Emp_Purchase2.Purchase==1],Emp_Purchase2.Experience[Emp_Purchase2.Purchase==1], s=10, c='r', marker="+", label='Purchase 1')
plt.xlim(min(Emp_Purchase2.Age), max(Emp_Purchase2.Age))
plt.ylim(min(Emp_Purchase2.Experience), max(Emp_Purchase2.Experience))
plt.legend(loc='upper left');

x_min, x_max = ax2.get_xlim()
y_min,y_max=ax2.get_ylim()
ax2.plot([x_min, x_max], [x_min*slope2+intercept2, x_max*slope2+intercept2])
plt.show()

#######Accuracy and error of the model1
#Create the confusion matrix
#Predciting Values
predicted_values=fitted2.predict(Emp_Purchase2[["Age"]+["Experience"]])
predicted_values[1:10]

#Lets convert them to classes using a threshold
threshold=0.5
threshold

import numpy as np
predicted_class=np.zeros(predicted_values.shape)
predicted_class[predicted_values>threshold]=1

#Predcited Classes
predicted_class[1:10]

from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Emp_Purchase2[['Purchase']],predicted_class)
print(ConfusionMatrix)
accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print(accuracy)

error=1-accuracy
error

###############################################
#### The Intermediate output and combined model
###############################################
#The Two models
fitted1.summary2()
fitted2.summary2()

#The two new coloumns
Emp_Purchase_raw['inter1']=fitted1.predict(Emp_Purchase_raw[["Age"]+["Experience"]])
Emp_Purchase_raw['inter2']=fitted2.predict(Emp_Purchase_raw[["Age"]+["Experience"]])

#plotting the new columns
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(Emp_Purchase_raw.inter1[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.inter2[Emp_Purchase_raw.Purchase==0], s=50, c='b', marker="o", label='Purchase 0')
ax.scatter(Emp_Purchase_raw.inter1[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.inter2[Emp_Purchase_raw.Purchase==1], s=50, c='r', marker="+", label='Purchase 1')

plt.xlim(min(Emp_Purchase_raw.inter1), max(Emp_Purchase_raw.inter1)+0.2)
plt.ylim(min(Emp_Purchase_raw.inter2), max(Emp_Purchase_raw.inter2)+0.2)

plt.legend(loc='lower left');
plt.show()

###Logistic Regerssion model with Intermediate outputs as input
import statsmodels.formula.api as sm

model_combined = sm.logit(formula='Purchase ~ inter1+inter2', data=Emp_Purchase_raw)
fitted_combined = model_combined.fit(method="bfgs")
fitted_combined.summary()

# getting slope and intercept of the line
slope_combined=fitted_combined.params[1]/(-fitted_combined.params[2])
intercept_combined=fitted_combined.params[0]/(-fitted_combined.params[2])

#Finally draw the decision boundary for this logistic regression model
import matplotlib.pyplot as plt

fig = plt.figure()
ax2 = fig.add_subplot(111)

ax2.scatter(Emp_Purchase_raw.inter1[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.inter2[Emp_Purchase_raw.Purchase==0], s=10, c='b', marker="o", label='Purchase 0')
ax2.scatter(Emp_Purchase_raw.inter1[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.inter2[Emp_Purchase_raw.Purchase==1], s=10, c='r', marker="+", label='Purchase 1')

plt.xlim(min(Emp_Purchase_raw.inter1), max(Emp_Purchase_raw.inter1)+0.2)
plt.ylim(min(Emp_Purchase_raw.inter2), max(Emp_Purchase_raw.inter2)+0.2)

plt.legend(loc='lower left');

x_min, x_max = ax2.get_xlim()
y_min,y_max=ax2.get_ylim()
ax2.plot([x_min, x_max], [x_min*slope_combined+intercept_combined, x_max*slope_combined+intercept_combined])
plt.show()

#######Accuracy and error of the model1
#Create the confusion matrix
#Predciting Values
predicted_values=fitted_combined.predict(Emp_Purchase_raw[["inter1"]+["inter2"]])
predicted_values[1:10]

#Lets convert them to classes using a threshold
threshold=0.5
threshold

import numpy as np
predicted_class=np.zeros(predicted_values.shape)
predicted_class[predicted_values>threshold]=1

#ConfusionMatrix
from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Emp_Purchase_raw[['Purchase']],predicted_class)
print(ConfusionMatrix)
accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print(accuracy)


###############################################
#### Building the neural network in Py ########  
###############################################

##############################LAB: Building the neural network in R####################
import pandas as pd
##Dataset: Emp_Purchase/Emp_Purchase.csv
Emp_Purchase_raw = pd.read_csv("./data/Emp_Purchase.csv")
Emp_Purchase_raw.shape
Emp_Purchase_raw.columns.values
Emp_Purchase_raw.head(10)

#Draw a scatter plot that shows Age on X axis and Experience on Y-axis. Try to distinguish the two classes with colors or shapes.
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==0], s=10, c='b', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==1], s=10, c='r', marker="+", label='Purchase 1')
plt.legend(loc='upper left');
plt.show()

####Building neural net
#We need to manually downlaod and install  neurolab use below command to install in Anaconda console
#pip install neurolab 
import neurolab as nl
import numpy as np
import pylab as pl

error = []
# Create network with 1 layer and random initialized

net = nl.net.newff([[15, 60],[1,20]],[2,1],transf=[nl.trans.LogSig()] * 2)
net.trainf = nl.train.train_rprop #Train algorithms based gradients algorithms - Resilient Backpropagation

#Number of input variables
net.ci

#Number of output variables
net.co

#Hidden layers + output layer
len(net.layers)


# Train network
net.train(Emp_Purchase_raw[["Age"]+["Experience"]], Emp_Purchase_raw[["Purchase"]], show=0, epochs = 500,goal=0.02)

#print(net.train(Emp_Purchase_raw[["Age"]+["Experience"]], Emp_Purchase_raw[["Purchase"]], show=0, epochs = 500,goal=0.02))
# Predicted values
predicted_values = net.sim(Emp_Purchase_raw[["Age"]+["Experience"]])

predicted_class=predicted_values
predicted_class[predicted_values>0.5]=1
predicted_class[predicted_values<=0.5]=0

#Predcited Classes
predicted_class[0:10]

#confusion matrix
from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Emp_Purchase_raw[['Purchase']],predicted_class)
print(ConfusionMatrix)

#accuracy
accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print(accuracy)

#plotting epoches Vs error
#pl.figure(1)
#pl.plot(error)
#pl.xlabel('Number of epochs')
#pl.ylabel('Training error')
#pl.grid()
#pl.show()

#plotting actual and prected classes
Emp_Purchase_raw['predicted_class']=pd.DataFrame(predicted_class)

import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==0], s=40, c='g', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==1], s=40, c='g', marker="x", label='Purchase 1')
ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.predicted_class==0],Emp_Purchase_raw.Experience[Emp_Purchase_raw.predicted_class==0], s=30, c='b', marker="1", label='Predicted 0')
ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.predicted_class==1],Emp_Purchase_raw.Experience[Emp_Purchase_raw.predicted_class==1], s=30, c='r', marker="2", label='Predicted 1')

plt.legend(loc='upper left');
plt.show()

#####

import matplotlib.pyplot as plt

x=plt.imread("./data/cat.jpg")
plt.imshow(x)

print("Shape of theimage ",x.shape)
print(x)

############################################
####Digit Recognition Example

################################LAB: Digit Recognizer################################

#Importing test and training data
import numpy as np
digits_train = np.loadtxt("./data/zip.train.txt")

#digits_train is numpy array. we convert it into dataframe for better handling
train_data=pd.DataFrame(digits_train)
train_data.shape

digits_test = np.loadtxt("./data//zip.test.txt")
#digits_test is numpy array. we convert it into dataframe for better handling
test_data=pd.DataFrame(digits_test)
test_data.shape

train_data[0].value_counts()     #To get labels of the images

import matplotlib.pyplot as plt

#Lets see some images.
#first image
data_row=digits_train[0][1:]
pixels = np.matrix(data_row)
pixels=pixels.reshape(16,16)
plt.imshow(pixels)

#second image
data_row=digits_train[1][1:]
pixels = np.matrix(data_row)
pixels=pixels.reshape(16,16)
plt.imshow(pixels)

#third image
data_row=digits_train[1000][1:]
pixels = np.matrix(data_row)
pixels=pixels.reshape(16,16)
plt.imshow(pixels)

#Creating multiple columns for multiple outputs
#####We need these variables while building the model
digit_labels=pd.DataFrame()
digit_labels['label']=train_data[0:][0]
label_names=['I0','I1','I2','I3','I4','I5','I6','I7','I8','I9']
for i in range(0,10):
	digit_labels[label_names[i]]=digit_labels.label==i

#see our newly created labels data
digit_labels.head(10)

#Update the training dataset
train_data1=pd.concat([train_data,digit_labels],axis=1)
train_data1.shape
train_data1.head(5)


#########Neural network building
import neurolab as nl
import numpy as np


x_train=train_data.drop(train_data.columns[[0]], axis=1)
y_train=digit_labels.drop(digit_labels.columns[[0]], axis=1)

#getting minimum and maximum of each column of x_train into a list
def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

listvalues = x_train.apply(minMax).T.values.tolist()

error = []
# Create network with 1 layer and random initialized
net = nl.net.newff(listvalues,[20,10],transf=[nl.trans.LogSig()] * 2)
net.trainf = nl.train.train_rprop #Training method is Resilient Backpropagation method 

# Train network
import time
start_time = time.time()
error.append(net.train(x_train, y_train, show=0, epochs = 250, goal=0.02))
print("--- %s seconds ---" % (time.time() - start_time))

# Prediction testing data
x_test=test_data.drop(test_data.columns[[0]], axis=1)
y_test=test_data[0:][0]

predicted_values = net.sim(x_test.as_matrix())
predict=pd.DataFrame(predicted_values)

index=predict.idxmax(axis=1)
#Predicted Values
index

#confusion matrix
from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(y_test,index)
print(ConfusionMatrix)

#accuracy
accuracy=np.trace(ConfusionMatrix)/sum(sum(ConfusionMatrix))
print(accuracy)

error=1-accuracy
print(error)








