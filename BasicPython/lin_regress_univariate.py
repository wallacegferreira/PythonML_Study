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

#
# Step 2: Provide data
#

# Simple - Univariate linear regression y = a0 + a1*x
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
print('coefficient of determination - R^2:', r_sq)

print('intercept - a0:', model.intercept_) # coefficient a0
print('slope - a1:', model.coef_) # coefficiet a1



