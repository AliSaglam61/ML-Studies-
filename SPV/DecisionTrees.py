# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:28:28 2021

@author: alisa
"""

from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
print('Sütun isimleri: {}'.format(bc.feature_names))
print('Sınıf İsimleri: {}'.format(bc.target_names))
dataset_x = bc.data
dataset_y = bc.target
from sklearn.model_selection import train_test_split
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(training_dataset_x, training_dataset_y)
score = dtc.score(test_dataset_x, test_dataset_y)
print(score)
import random
r = random.randint(0, len(test_dataset_x) - 1)
result = dtc.predict(test_dataset_x[r].reshape((1, -1)))
print(bc.target_names[result[0]])
print(test_dataset_y[r] == result[0])
from sklearn.tree import export_graphviz
import graphviz
export_graphviz(dtc, out_file='dt.gv', class_names=['malignant', 'benign'],
feature_names=bc.feature_names,impurity=False, filled=True)
g = graphviz.Source.from_file('dt.gv')