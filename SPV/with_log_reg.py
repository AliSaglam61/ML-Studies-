# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:32:30 2021

@author: alisa
"""

from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
print('Sütun isimleri: {}'.format(bc.feature_names))
print('Sınıf İsimleri: {}'.format(bc.target_names))
dataset_x = bc.data
dataset_y = bc.target
from sklearn.model_selection import train_test_split
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(dataset_x, dataset_y)
score = lr.score(test_dataset_x, test_dataset_y)
print(score)
result = lr.predict(test_dataset_x)
ratio = sum(result == test_dataset_y) / len(test_dataset_y)
print(ratio)