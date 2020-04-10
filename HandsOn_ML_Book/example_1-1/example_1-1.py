# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:18:13 2020

@author: Wallace

Code adapted from: 
    
"Hands On Machine Learning" 2nd Edition, by Aurélien Géron
https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# Data frame processing
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]


# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)
r_sq = model.score(X, y)
b0 = model.intercept_
b1 = model.coef_[0]
y_pred = model.predict(X)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
y_new = model.predict(X_new)
print(y_new) # outputs [[ 5.96242338]]


line = "Fit - R^2 ={:6.3f}".format(r_sq)

fig, ax = plt.subplots()
ax.plot(X, y, linewidth=0, marker='s', label='Data points')
ax.plot(X, b0 + b1*X, label=line)
ax.plot(X_new,y_new, color='red', marker='*', linewidth=0, markersize=12, label='Cyprus estimate')
ax.set_xlabel('GPD per capita')
ax.set_ylabel('Life Satisfaction')
ax.legend(facecolor='white')
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(y, y_pred, linewidth=0, marker='s', label = "Predicted")
ax2.plot(y, y, label=' Exact:  y_pred = y')
ax2.set_xlabel('Actual - Life Satisfaction')
ax2.set_ylabel('Predicted - Life Satisfaction')
ax2.legend(facecolor='white')
plt.show()


