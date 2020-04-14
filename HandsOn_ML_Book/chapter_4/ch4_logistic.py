# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:19:16 2020

@author: Wallace

Based (with some adaptations) on:

"Hands On Machine Learning" 2nd Edition, 
by Aurélien Géron https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris()

X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0
print(y)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42,C=10**0)
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

#plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
#plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")


X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]



plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()


print('Model Accuracy: \n')
print(log_reg.score(X,y))
print('\n')

print('Confusion Matrix:')
print(confusion_matrix(y, log_reg.predict(X)))
print('\n')

print(classification_report(y, log_reg.predict(X)))



b0 = log_reg.intercept_
b1 = log_reg.coef_[0]
y_log_reg = b0 + b1*X # Logit function: y = b0 + b1*x


from scipy.special import expit
xt = np.linspace(0,10,100)
pt = expit(b0+ b1*xt)


fig, ax = plt.subplots()
plt.plot(X[y==0], y[y==0], "bs",label='Not Virginica')
plt.plot(X[y==1], y[y==1], "g^",label='Virginica')
plt.plot(X,y_log_reg, 'r-',label='Regr. Line, f(x)')  
plt.axis([0, 3, -0.02, 1.02])
plt.plot(xt,pt, 'k-',label='Logit Line, p(x)')
plt.legend(facecolor='white')
plt.show()