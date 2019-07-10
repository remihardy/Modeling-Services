# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:25:01 2019

@author: HARDY
"""

#-------------------------------------
#2 features (continuous values) / 3 outputs (continuous values)
#PURPOSE:
#- provide a machine learning based model (rather than parametric)
#- determine the minimum number of points used for training, while keeping an acceptable accuracy 
#-------------------------------------



#-------------------------------------
#PARAMETERS
#-------------------------------------

FILE_PATH='#####'
TEST_SIZE = 0.30 #train 70% test 30%
#can be used to have a well defined bins list for the histogram
BINS_LIST=[0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

#-------------------------------------
#LIBS
#-------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import math
from math import modf


#-------------------------------------
#load data set and add a header
#-------------------------------------

df = pd.read_csv(FILE_PATH,sep=' ', names=["x","y","p1","p2","p3"])
#selecting x and y as inputs
X=df.iloc[:,0:2]
#selecting p1 p2 p3 as outputs
y=df.iloc[:,2:5]


#-------------------------------------
#check the distribution of x and y coordinates
#-------------------------------------
x_flat=np.array(X['x'])
x_flat=x_flat.tolist()
plt.hist(x_flat, bins=25, color='green')
plt.xlabel('x coordinates distribution')
plt.legend()
plt.show()
plt.clf()
y_flat=np.array(X['y'])
y_flat=y_flat.tolist()
plt.hist(y_flat, bins=25, color='red')
plt.xlabel('y coordinates distribution')
plt.show()

#-------------------------------------
#split dataset to train and validation 
#-------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1234)
 

#-------------------------------------
#Feature scaling
#-------------------------------------
#Feature Scaling (x-u)/s removing mean and scaling to unit variance
#has a significant effect eventhough X Y Pi are (relatively) on the same scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#--------------------------
#MODEL
#-------------------------
#activation and solver choice have a significant impact
mymodel=MLPRegressor(hidden_layer_sizes=(150),activation='tanh', solver='lbfgs', alpha=0.001, batch_size='auto',learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,random_state=9, tol=0.0001, verbose=False, warm_start=True, momentum=0.9, nesterovs_momentum=True,early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
mymodel.fit(X_train,y_train) 
y_pred=mymodel.predict(X_test)      
y_pred=y_pred.reshape(y_pred.shape[0]*y_pred.shape[1])
y_true=np.array(y_test)
y_true=y_true.reshape(y_true.shape[0]*y_true.shape[1])


#--------------------------
#Round y_pred to nearest integer due to PWM mechanical constraints
#-------------------------

for i in range(len(y_pred)):
    if math.modf(y_pred[i])[0]>0.5:
        y_pred[i]=math.ceil(y_pred[i])
    else:
        y_pred[i]=math.floor(y_pred[i])        

#--------------------------
#METRICS
#-------------------------
print ('Number of points used for training : ' + str(len(X_train)))
print ('Number of points used for test     : ' + str(len(X_test)))
#R2 score
R2=mymodel.score(X_test,y_test) 
print('R2 score = ' + str(R2))
#compute sum of squared errors
sse=((y_pred-y_true)**2).sum() 
#absolute error (vector)
error=np.abs(y_true - y_pred)
#absolute percent error (vector)
aperr= np.abs((y_true - y_pred) / y_true)
#MAPE 
mape=np.mean(np.abs((y_true - y_pred) / y_true)) 
print('MAPE = ' + str(mape))


#--------------------------
#PLOT
#-------------------------
#plot the absolute percentage error distribution

plt.hist(aperr, bins=25, color='blue')
#plt.hist(aperr, bins=BINS_LIST, color='blue')
plt.xlabel('Absolute Percentage Error')
plt.show()



