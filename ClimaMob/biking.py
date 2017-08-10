# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:35:35 2017

@author: mandart
"""

import numpy as np 
import pandas as pd 
import csv
from io import StringIO
import io
#from StringIO import StringIO
#from sklearn import cross_validation, grid_search, linear_model, metrics, pipeline, preprocessing
from numpy import genfromtxt
datapd = pd.read_csv('C:\\Users\\mandart\\Documents\\GitHub\\MachineLearning\\MachineLearning\\Regression\\bikedata21.csv',sep=';',dtype=None,header=0)
print(datapd.head())
print(datapd.dtypes)
print(datapd.describe())

X=datapd.iloc[:,0]
indices = [i for i, x in enumerate(X) if x == 300016]
X4=datapd.iloc[indices,6]
indexword=X4.str.contains('HOURLY')
lindexword=[i for i, x in enumerate(indexword) if x]
X5=datapd.iloc[indices,:9]
X6=X5.iloc[lindexword,:]
X7=X6.iloc[:,5]
indices1 = [i for i, x in enumerate(X7) if x == 1]
S300016=X6.iloc[indices1,:]
S300016.to_csv('S300016_1.csv', index=False, header=False)
indices2 = [i for i, x in enumerate(X7) if x == 2]
S3000162=X6.iloc[indices2,:]
S3000162.to_csv('300016_2.csv', index=False, header=False)
del X5,X,X4,X6,X7


X=datapd.iloc[:,0]
indices = [i for i, x in enumerate(X) if x == 300083]
X4=datapd.iloc[indices,6]
indexword=X4.str.contains('HOURLY')
lindexword=[i for i, x in enumerate(indexword) if x]
X5=datapd.iloc[indices,:9]
X6=X5.iloc[lindexword,:]
X7=X6.iloc[:,5]
indices1 = [i for i, x in enumerate(X7) if x == 1]
S300083=X6.iloc[indices1,:]
S300083.to_csv('S300083_1.csv', index=False, header=False)
indices2 = [i for i, x in enumerate(X7) if x == 2]
S3000832=X6.iloc[indices2,:]
S3000832.to_csv('S300083_2.csv', index=False, header=False)
del X5,X,X4,X6,X7

X=datapd.iloc[:,0]
indices = [i for i, x in enumerate(X) if x == 300099]
X4=datapd.iloc[indices,6]
indexword=X4.str.contains('HOURLY')
lindexword=[i for i, x in enumerate(indexword) if x]
X5=datapd.iloc[indices,:9]
X6=X5.iloc[lindexword,:]
X7=X6.iloc[:,5]
indices1 = [i for i, x in enumerate(X7) if x == 1]
S300099=X6.iloc[indices1,:]
S300099.to_csv('300099_1.csv', index=False, header=False)
indices2 = [i for i, x in enumerate(X7) if x == 2]
S3000992=X6.iloc[indices2,:]
S3000992.to_csv('300099_2.csv', index=False, header=False)
del X5,X,X4,X6,X7

X=datapd.iloc[:,0]
indices = [i for i, x in enumerate(X) if x == 300233]
X4=datapd.iloc[indices,6]
indexword=X4.str.contains('HOURLY')
lindexword=[i for i, x in enumerate(indexword) if x]
X5=datapd.iloc[indices,:9]
X6=X5.iloc[lindexword,:]
X7=X6.iloc[:,5]
indices1 = [i for i, x in enumerate(X7) if x == 1]
S3002331=X6.iloc[indices1,:]
S3002331.to_csv('S300233_1.csv', index=False, header=False)
indices2 = [i for i, x in enumerate(X7) if x == 2]
S3002332=X6.iloc[indices2,:]
S3002332.to_csv('S300233_2.csv', index=False, header=False)
del X5,X,X4,X6,X7

X=datapd.iloc[:,0]
indices = [i for i, x in enumerate(X) if x == 302257]
X4=datapd.iloc[indices,6]
indexword=X4.str.contains('HOURLY')
lindexword=[i for i, x in enumerate(indexword) if x]
X5=datapd.iloc[indices,:9]
X6=X5.iloc[lindexword,:]
X7=X6.iloc[:,5]
indices1 = [i for i, x in enumerate(X7) if x == 1]
S3022571=X6.iloc[indices1,:]
S3022571.to_csv('S302257_1.csv', index=False, header=False)
indices2 = [i for i, x in enumerate(X7) if x == 2]
S3022572=X6.iloc[indices2,:]
S3022572.to_csv('S302257_2.csv', index=False, header=False)
del X5,X,X4,X6,X7

datapd2 = pd.read_csv('C:\\Users\\mandart\\Documents\\GitHub\\MachineLearning\\MachineLearning\\Regression\\Oslo_Blindern_Weather.csv',sep=';',dtype=None,header=0)
datapd2.to_csv('weather_osloMET.csv', index=False, header=False)
xx=datapd2.iloc[:8015,5:10]
print(xx.dtypes)
xx = xx.set_index(S300099.index)
#catS300099=pd.concat(['FieldName',xx], axis=1)
S300099['DirectionDeg']=xx.iloc[:,0].values
S300099['Speedmps']=xx.iloc[:,1].values
S300099['TempC']=xx.iloc[:,2].values
S300099['Pptmm']=xx.iloc[:,3].values 
S300099['Humidpercen']=xx.iloc[:,4].values 
S300099.to_csv('Dataset1.csv', index=False, header=True)     

S300099.plot(kind='scatter', x='Speedmps', y='SUM', figsize=(12,8))

#Normalizing the data
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

#S300099norm = (S300099 - S300099.mean()) / S300099.std()
#
##X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state = 2)
#
#
#from sklearn import linear_model
#model = linear_model.LinearRegression()
#model.fit(X, y)

  
#xx['DirectionDeg']
#.astype(int64)
#	Speedmps	TempC	Pptmm	Humidpercen]
#S3000992


#df2 = pd.DataFrame(X5, index=index, columns=columns)
#~datapd["RESOLUTION"]

#indicesx = np.where(X4 == 'HOURLY')
#
#print (indicesx) 
#
#print (X4[indicesx])

#datapd.iloc[6,indicesx].index.tolist()
#X5=data[indices,5]
    
#with open('outputbikedata2x.csv', 'w') as outfile:
#    writer = csv.writer(outfile)
#    outfile.write('Col1name, Col2name, Col3name')
#    for row in zip(Stationnumber1,Time1,Count1):
#        writer.writerow(row)
    
#np.savetxt("bikeextract1.csv", X1, delimiter=",")

#ofile  = open('bikeextracttest.csv', "wb")
#writer = csv.writer(ofile, delimiter='', quotechar='', quoting=csv.QUOTE_ALL)
# 
#for row in X1:
#    writer.writerow(row)
# 
#ofile.close()

#X = np.array(X.values)
#y = np.matrix(y.values)
#theta = np.matrix(np.array([0,0]))


#y = data[:,1]
#m1 = len(X)
#print('m1', m1)
#m = len(indices)
#print('m', m)
#mx = len(INDEXWORD)
#print('mx', mx)