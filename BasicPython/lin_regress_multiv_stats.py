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
import statsmodels.api as sm
import matplotlib.pyplot as plt


#
# Step 2: Provide data
#

# Multiple - Multivariate linear regression y = b0 + b11*x1 + b12*x2  +... + b1n*xn
# Vector of independent variable
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

x = sm.add_constant(x) #intercept is not added by default


#
# Step 3: Create a model and fit it
#

model = sm.OLS(y, x)


#
# Step 4: Get results
#

results = model.fit()
print(results.summary())

r_sq = results.rsquared
print('Coefficient of determination - R^2:', r_sq)

b0 = results.params[0]
b11 = results.params[1]
b12 = results.params[2]

print('Intercept - b0:', b0) # coefficient b0
print('Slopes - b11 and b12:', b11,b12) # coefficients b11 and b12

#
# Step 5: Predict response
# 

y_pred = results.fittedvalues # saeme of results.predict(x)
print('Predicted response:', y_pred, sep='\n')


# Predicting for new data
x_new = sm.add_constant(np.arange(10).reshape((-1, 2)))
print('New data points:')
print(x_new)

y_new = results.predict(x_new)
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



