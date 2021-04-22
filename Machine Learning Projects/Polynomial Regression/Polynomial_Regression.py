# Polynomial Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Position_Salaries.csv')
# when trying to plot data it appears to be part of parapola
xplot = np.array(data['Level'])
yplot= np.array(data['Salary'])
plt.scatter(xplot,yplot)

# we will try tp predict Salary upon Level 
X = data.iloc[:,1].values.reshape(-1,1) # x must be a matrix so we reshapes it 
y = data.iloc[:,2].values

# we will not split our data coz it is so little and we need high accuracy
# we will make linear regressin to compare it's results with polynomial model
# fiiting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict=lin_reg.predict(X)
# fitting polynomial refression to dataset with dgree 2
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)

# Now we will create another instance of LinearRegression and fit the x_poly matrix
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
y_pred_poly=lin_reg2.predict(X_poly)
# Visualizing result for linear regression model 
# step 1 visulaize real observations
plt.scatter(X,y, color='red')
#step 2 visualize predicted values
plt.plot(X,y_predict, color='green')
plt.title('Linear regression model')
plt.xlabel('Level')
plt.ylabel('Salalry')
# as we saw from visualized results it's far a way from being trusted model

# Visualizing polynomial model
plt.scatter(X,y, color='green')
#step 2 visualize predicted values
plt.plot(X,y_pred_poly, color='blue')
plt.title('polynomial regression model')
plt.xlabel('Level')
plt.ylabel('Salalry')
# as we can see from our plot it's more acurate than linear regression
# fitting polynomial refression to dataset with dgree 3
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X)

# Now we will create another instance of LinearRegression and fit the x_poly matrix
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
y_pred_poly=lin_reg2.predict(X_poly)