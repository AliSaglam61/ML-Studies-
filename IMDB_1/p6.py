# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:36:26 2021

@author: alisa
"""

import numpy as np
import re
class WordConverter:
    def __init__(self):
        self.word_set = set()
    def __call__(self, text):
        words = re.findall("[a-zA-Z0-9'-]+", text.lower())
        self.word_set.update(words)
        return np.array(words)
    def get_word_dict(self):
        return {word: index for index, word in enumerate(wc.word_set)}
import pandas as pd
wc = WordConverter()
df = pd.read_csv('IMDB Dataset.csv', converters={0: wc})
word_dict = wc.get_word_dict()
for i in range(len(df)):
    df.iloc[i, 0] = np.array([word_dict[word] for word in df.iloc[i, 0]])
def vectorize(iterable, colsize):
    result = np.zeros((len(iterable), colsize), dtype=np.int8)
    for index, vals in enumerate(iterable):
        result[index, vals] = 1
        return result    
dataset = df['review'].to_list()
dataset_x = vectorize(dataset, len(word_dict))
df.iloc[df.iloc[:, 1] == 'positive', 1] = 1
df.iloc[df.iloc[:, 1] == 'negative', 1] = 0
dataset_y = df.iloc[:, 1].to_numpy(dtype=np.int8)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y,
test_size=0.20)
model = Sequential(name='IMDB')
model.add(Dense(100, input_dim=training_dataset_x.shape[1], activation='relu', name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=64, epochs=1,
validation_split=0.2)
import matplotlib.pyplot as plt
figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()    
figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Binary Accuracy - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Binary Accuracy')
plt.plot(range(1, len(hist.history['binary_accuracy']) + 1), hist.history['binary_accuracy'])
plt.plot(range(1, len(hist.history['val_binary_accuracy']) + 1),
hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()
eval_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
predict_text = "A bad production. It's a bad movie."

def prepare_review(predict_text):
    predict_words = re.findall("[a-zA-Z0-9'-]+", predict_text.lower())
    result = np.zeros(len(predict_words), dtype=np.int8)
    for i in range(len(predict_words)):
        result[i] = word_dict[predict_words[i]]
    return vectorize([result], len(word_dict))
predict_data = prepare_review(predict_text)
predict_result = model.predict(predict_data)
if predict_result[0, 0] > 0.5:
    print(f'Olumlu Yorum: {predict_result[0, 0]}')
else:
    print(f'Olumsuz Yorum: {predict_result[0, 0]}')     