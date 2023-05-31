# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:35:19 2021

@author: alisa
"""

from sklearn.datasets import make_circles
dataset_x, dataset_y = make_circles(n_samples=100)
from sklearn.model_selection import train_test_split
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.3)
import matplotlib.pyplot as plt
plt.scatter(training_dataset_x[training_dataset_y == 0, 0],
training_dataset_x[training_dataset_y == 0, 1], color='blue')
plt.title('Radial Random Points')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(training_dataset_x[training_dataset_y == 1, 0],
training_dataset_x[training_dataset_y == 1, 1], color='green')
plt.pause(1)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(training_dataset_x, training_dataset_y)
result = svc.predict(test_dataset_x)
ratio = sum(result == test_dataset_y) / len(test_dataset_y)
print('Başarı oranı: {}'.format(ratio))
print('Score: {}'.format(svc.score(training_dataset_x, training_dataset_y)))