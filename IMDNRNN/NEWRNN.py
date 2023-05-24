# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 22:36:55 2021

@author: alisa
"""

from tensorflow.keras.datasets import imdb
max_features = 10000
max_text_len = 500
batch_size = 512
epochs = 5
(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) =imdb.load_data(num_words=max_features)
from tensorflow.keras.preprocessing import sequence
training_dataset_x = sequence.pad_sequences(training_dataset_x, maxlen=max_text_len)
test_dataset_x = sequence.pad_sequences(test_dataset_x, maxlen=max_text_len)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(64, return_sequences=True, activation='relu'))
model.add(SimpleRNN(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(training_dataset_x, training_dataset_y, epochs=epochs,
batch_size=batch_size, validation_split=0.2)
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(range(1, epochs + 1), acc, 'r', label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_acc, 'b', label='Validation Accuracy')
plt.legend(['acc', 'val_acc'])
plt.pause(1)
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(1, epochs + 1), loss, 'r', label='Training Loss')
plt.plot(range(1, epochs + 1), val_loss, 'b', label='Validation Loss')
plt.legend(['loss', 'val_loss'])
