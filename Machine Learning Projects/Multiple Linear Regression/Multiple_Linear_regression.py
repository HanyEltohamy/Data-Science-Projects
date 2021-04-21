# -*- coding: utf-8 -*-
"""
@author: Hany Eltohamy
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
data = pd.read_csv('50_Startups.csv')
X = data.iloc[:,:-1].values
y=data.iloc[:,4]

# first we need to encode state columns column 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer

label_encoder_x = LabelEncoder()
X[: , 3] = label_encoder_x.fit_transform(X[:,3])
columntransformer = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = columntransformer.fit_transform(X)
# avoid dummy variable trap
X=X[:,1:]

# Splitting our data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# fitting multiple linear regression to train set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)
# Predicting results 
y_predict= regressor.predict(X_test)
# Building model using backward elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt= X[:,[0,1,2,3,4,5]].astype(float)
regressor_OLS=sm.OLS(endog=y.astype(float),exog=X_opt).fit()

print(regressor_OLS.summary())
# p value of x2 is higher than alpha so we will remove it 
X_opt= X[:,[0,1,3,4,5]].astype(float)
regressor_OLS=sm.OLS(endog=y.astype(float),exog=X_opt).fit()
print(regressor_OLS.summary())

# p value of x1 is higher than alpha so we will remove it 
X_opt= X[:,[0,3,4,5]].astype(float)
regressor_OLS=sm.OLS(endog=y.astype(float),exog=X_opt).fit()
print(regressor_OLS.summary())
# p value of x2 is higher than alpha so we will remove it 
X_opt= X[:,[0,3,5]].astype(float)
regressor_OLS=sm.OLS(endog=y.astype(float),exog=X_opt).fit()
print(regressor_OLS.summary())
# p value of x2 is higher than alpha so we will remove it 
X_opt= X[:,[0,3]].astype(float)
regressor_OLS=sm.OLS(endog=y.astype(float),exog=X_opt).fit()
print(regressor_OLS.summary())


"""
conclusion if we are looking for optimal team of independant variables that 
can predict profit with high statistical significance 
is actually commposed of one independant variable which is R&D spend
"""











