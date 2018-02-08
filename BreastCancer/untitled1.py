# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 00:16:15 2018

@author: Rachit
"""

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Activation,Layer,Lambda

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("data.csv")

dataset.drop(["id","Unnamed: 32"],axis=1,inplace=True)

dataset["diagnosis"] = dataset["diagnosis"].map({'M':1,'B':0})

y = dataset["diagnosis"]
X = dataset.drop(["diagnosis"],axis=1)

#dividing data set in training, CV and test set

trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.2, random_state = 42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size = 0.2, random_state = 42)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)

model = Sequential()
model.add(Dense(input_dim=30,output_dim=50,init="uniform",activation="relu"))
model.add(Dense(input_dim=50,output_dim=40,activation="relu"))
model.add(Dense(input_dim=40,output_dim=1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

scaler = StandardScaler()
model.fit(scaler.fit_transform(trainX), trainY,epochs=10,batch_size=20)

testN = scaler.transform(testX)
predY = model.predict(testN)
predY = (predY>0.5)
from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(testY,predY)
acc = accuracy_score(predY,np.array(testY))
print(cm)
print(acc)