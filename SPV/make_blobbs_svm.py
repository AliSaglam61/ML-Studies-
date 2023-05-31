# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:33:18 2021

@author: alisa
"""

import numpy as np
from sklearn.datasets import make_blobs
dataset_x, dataset_y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.8)
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(dataset_x, dataset_y)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(dataset_x, dataset_y)
import matplotlib.pyplot as plt
plt.title('Comparison SVM and LR Lines')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(dataset_x[:, 0], dataset_x[:, 1], c=dataset_y)
x = np.linspace(-10, 10, 100)
y = (-svc.coef_[0][0] * x - svc.intercept_[0]) / svc.coef_[0][1]
plt.plot(x, y, c='red')
y = (-lr.coef_[0][0] * x - lr.intercept_[0]) / lr.coef_[0][1]
plt.plot(x, y, c='blue')