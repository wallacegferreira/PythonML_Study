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
# alternatively: scipy.stats.linregress()
import matplotlib.pyplot as plt

#
# Step 2: Provide data
#

# Simple - Univariate linear regression y = b0 + b1*x
# Vector of independent variable
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# or instead x = np.array([[5], [15], [25], [35], [45], [55]])

# Vector of dependent variable
y = np.array([5, 20, 14, 32, 22, 38])


#
# Step 3: Create a model and fit it
#

model = LinearRegression()
model.fit(x, y)
# or instead model = LinearRegression().fit(x, y)


#
# Step 4: Get results
#

r_sq = model.score(x, y)
print('Coefficient of determination - R^2:', r_sq)

print('Intercept - b0:', model.intercept_) # coefficient b0
print('Slope - b1:', model.coef_) # coefficient b1

#
# Step 5: Predict response
# 

y_pred = model.predict(x)  # y_pred = model.intercept_ + model.coef_ * x
print('Predicted response:', y_pred, sep='\n')


# Predicting for new data
x_new = np.arange(5).reshape((-1, 1))
print('New data points:')
print(x_new)

y_new = model.predict(x_new)
print('Predicted response for new data:')
print(y_new)


#
# Step 6: Plot data
# 

b0 = model.intercept_
b1 = model.coef_[0]


line = "Regression line: y ={:6.2f} +{:6.2f}x - R^2 ={:6.2f}".format(b0,b1,r_sq)
print(line)
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, b0 + b1*x, label=line)
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



