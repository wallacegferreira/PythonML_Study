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
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#
# Step 2: Provide data
#

#A set of binary data
x, y = load_digits(return_X_y=True)


#
# Step 2b: Split data
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


#
# Step 2c: Scale data
#
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)


#
# Step 3: Create a model and fit it
#
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', random_state=0)
model.fit(x_train, y_train)


#
# Step 4: Evaluate the model
#
x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)

print('Model Scor for Training Set:\n')
print(model.score(x_train, y_train))
print('\n')
print('Model Scor for Test Set:\n')
print(model.score(x_test, y_test))
print('\n')

#Print Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
ax.set_ylabel('Actual outputs', fontsize=12, color='black')
ax.xaxis.set(ticks=range(10))
ax.yaxis.set(ticks=range(10))
ax.set_title('Confusion Matrix', fontsize=12, color='black')
ax.set_ylim(9.5, -0.5)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()

