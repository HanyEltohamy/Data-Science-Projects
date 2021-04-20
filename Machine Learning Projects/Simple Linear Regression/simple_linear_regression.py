# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:13:54 2021

@author: Hany Eltohamy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Salary_Data.csv')

x = data['YearsExperience'].values.reshape(-1,1)
y = data['Salary'].values

# splitting data into tarin and test 
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(x,y,test_size=1/3,random_state=0)
# fitting simple le=inear regreesion to trainng set

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#Predicting values for x_test
y_predict= regressor.predict(X_test)

plt.scatter(X_train, y_train , marker='o' , color='red')
plt.scatter(X_test, y_test , marker='o' , color='black')
plt.plot(X_train,regressor.predict(X_train))
plt.xlabel('Years of Experince')
plt.ylabel('salary')
plt.title('Predicting Salary')

print(regressor.coef_)
print(regressor.intercept_)
