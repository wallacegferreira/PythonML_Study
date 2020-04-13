# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:05:29 2020

@author: Wallace

Examples based on https://realpython.com/logistic-regression-python/
"""
#
# Step 1: Import packages and classes
#
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import expit


#
# Step 2: Provide data
#

#A set of binary data
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

fig, ax = plt.subplots()
ax.plot(x[:4], y[:4], linewidth=0, marker='o', label='Class 0')
ax.plot(x[4:], y[4:], linewidth=0, marker='v', label='Class 1')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()


#
# Step 3: Create a model and fit it
#

model = LogisticRegression(solver='liblinear', random_state=0)

model.fit(x, y)

# Or instead: model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)

print('Model Classes: \n')
print(model.classes_)
print(model.intercept_)
print('\n')
print('Model Coefficient: \n')
print(model.coef_)
print('\n')


#
# Step 4: Evaluate the model
#
print('Model Probabilities: \n')
print(model.predict_proba(x))
print('\n')


#
# Step 5: Plot data
# 

b0 = model.intercept_
b1 = model.coef_[0]
y_pred = b0 + b1*x # Logit function: y = b0 + b1*x
y_pred_prob = expit(y_pred) # Sigmoid function: p = 1/(1 + exp(-y))

#Sigmoid function
xt = np.linspace(0,10,100)
pt = expit(b0+ b1*xt)


fig, ax = plt.subplots()
ax.plot(x[:4], y[:4], linewidth=0, marker='o', label='Class 0')
ax.plot(x[4:], y[4:], linewidth=0, marker='v', label='Class 1')
ax.plot(x,y_pred, 'k--', label='y(x): Regression Line')
ax.plot(x, y_pred_prob, linewidth=0, marker='+', markersize=10, label='Predicted Probability')
ax.plot(xt,pt, 'r-', label='p(y(x)): Probability line')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.ylim(-0.25, 1.25)
ax.legend(facecolor='white')
plt.show()


#
# Step 6: Evaluate accuracy
#
print('Model Accuracy: \n')
print(model.score(x,y))
print('\n')

print('Confusion Matrix:')
print(confusion_matrix(y, model.predict(x)))
print('\n')


#Plot Confusion Matrix
cm = confusion_matrix(y, model.predict(x))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


print(classification_report(y, model.predict(x)))