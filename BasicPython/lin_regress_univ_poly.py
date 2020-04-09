# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:05:29 2020

@author: Wallace

Examples based on https://realpython.com/linear-regression-in-python/
"""
#
# Step 1: Import packages and classes
#
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#
# Step 2a: Provide data
#

# Simple - Univariate polynomia linear regression y = b0 + b1*x b2*x^2
# Vector of independent variable
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

#
# Step 2b: Transfor data for polynomial fit of higher order terms
#
transformer = PolynomialFeatures(degree=2, include_bias=False)
# include_bias = False -> neglect intercept in the model
transformer.fit(x)
x_ = transformer.transform(x)
# fit_transform() can be used at once to fit and trasnform


#
# Step 3: Create a model and fit it
#

model = LinearRegression().fit(x_, y)

# Alternatively the model intercept should be created by using:
# x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
# model = LinearRegression(fit_intercept=False).fit(x_, y)

#
# Step 4: Get results
#

r_sq = model.score(x_, y)
print('Coefficient of determination - R^2:', r_sq)

print('Intercept - b0:', model.intercept_) # coefficient b0
print('Slope - b11 and b12:', model.coef_) # coefficient b1

#
# Step 5: Predict response
# 

y_pred = model.predict(x_)  # 
print('Predicted response:', y_pred, sep='\n')



#
# Step 6: Plot data
# 

b0 = model.intercept_
b11 = model.coef_[0]
b12 = model.coef_[1]


line = "Regression: y ={:6.2f} +{:6.2f}x +{:6.2f}x^2 \n\nR^2 ={:6.2f}".format(b0,b11,b12,r_sq)
print(line)
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, b0 + b11*x + b12*x**2, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(y, y_pred, linewidth=0, marker='s', label = "Predicted")
ax2.plot(y, y, label=' Exact, y_pred = y')
ax2.set_xlabel('Actual Response, y')
ax2.set_ylabel('Predicted Response, y_pred')
ax2.legend(facecolor='white')
plt.show()



