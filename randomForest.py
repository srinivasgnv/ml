import pandas as pd
import sklearn as sk
import numpy as np
import scipy as sp

#Importing dataset
car_train=pd.read_csv("./data/train.csv")
car_test=pd.read_csv("./data/test.csv")

from sklearn import tree

var=list(car_train.columns[1:22])
c=car_train[var]
d=car_train['Fatal']

###buildng Decision tree on the training data ####
clf = tree.DecisionTreeClassifier()
clf.fit(c,d)

#####predicting on test data ####
tree_predict=clf.predict(car_test[var])

from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm1 = confusion_matrix(car_test[['Fatal']],tree_predict)
print(cm1)

#####from confusion matrix calculate accuracy
total1=sum(sum(cm1))
accuracy_tree=(cm1[0,0]+cm1[1,1])/total1
accuracy_tree


####Building Random Forest Model
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=10, max_features=5)

forest.fit(c,d)

forestpredict_test=forest.predict(car_test[var])
e=car_test['Fatal']

###check the accuracy on test data
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm2 = confusion_matrix(car_test[['Fatal']],forestpredict_test)
print(cm2)
total2=sum(sum(cm2))
#####from confusion matrix calculate accuracy
accuracy_forest=(cm2[0,0]+cm2[1,1])/total2
accuracy_forest

