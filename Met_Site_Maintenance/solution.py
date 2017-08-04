# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:40:41 2017

@author: arash
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
import numpy as np

training_data=pd.read_csv('Arundo_take_home_challenge_training_set.csv',sep=',',parse_dates=['date'])
training_data['day_of_week'] = training_data['date'].dt.dayofweek
training_data['events_code'] = pd.Categorical(training_data["events"]).codes
y=training_data["request_count"]
training_data = training_data.drop(["date","events","request_count"],axis=1)
#One hot encoding of the categorical variables
training_data= pd.get_dummies(training_data,columns=[calendar_code","events_code","day_of_week"],prefix=["calendar","event","week"])

X=training_data.values
X_train, X_val, y_train, y_val =  train_test_split(X,y,test_size=0.2,random_state = 0)

#Multivariabte regression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_train_pred=regr.predict(X_train)
print("Mean squared error: %.2f" % np.mean((regr.predict(X_train) - y_train) ** 2))
print("Mean squared error: %.2f" % np.mean((regr.predict(X_val) - y_val) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_train, y_train))
print('Variance score: %.2f' % regr.score(X_val, y_val))

#Support Vector Machine
svc = svm.SVC(kernel='linear', gamma=0.01, C=0.1).fit(X_train, y_train)
print("Mean squared error: %.2f" % np.mean((svc.predict(X_train) - y_train) ** 2))
print("Mean squared error: %.2f" % np.mean((svc.predict(X_val) - y_val) ** 2))