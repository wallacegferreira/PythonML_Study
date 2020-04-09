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

# Multiple - Multivariate linear regression y = b0 + b11*x1 + b12*x2  +... + b1n*xn
# Vector of independent variable
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)


#
# Step 3: Create a model and fit it
#

model = LinearRegression().fit(x, y)


#
# Step 4: Get results
#

r_sq = model.score(x, y)
print('Coefficient of determination - R^2:', r_sq)

print('Intercept - b0:', model.intercept_) # coefficient b0
print('Slopes - b11 and b12:', model.coef_) # coefficients b11 and b12

#
# Step 5: Predict response
# 

y_pred = model.predict(x)  # y_pred = model.intercept_ + model.coef_[0] * x[0] +  model.coef_[1] * x[1]
print('Predicted response:', y_pred, sep='\n')


# Predicting for new data
x_new = np.arange(10).reshape((-1, 2))
print('New data points:')
print(x_new)

y_new = model.predict(x_new)
print('Predicted response for new data:')
print(y_new)

#
# Step 6: Plot data
# 


fig, ax = plt.subplots()
ax.plot(y, y_pred, linewidth=0, marker='s', label = "Predicted")
ax.plot(y, y, label=' Exact, y_pred = y')
ax.set_xlabel('Actual Response, y')
ax.set_ylabel('Predicted Response, y_pred')
ax.legend(facecolor='white')
plt.show()



