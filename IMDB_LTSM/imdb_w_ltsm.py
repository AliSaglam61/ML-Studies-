# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:24:33 2023

@author: alisa
"""

VOCAB_SIZE = 30000
TEXT_SIZE = 300
from tensorflow.keras.datasets import imdb
(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = imdb.load_data(num_words=VOCAB_SIZE)
word_dict = imdb.get_word_index()
from tensorflow.keras.preprocessing.sequence import pad_sequences
training_dataset_x = pad_sequences(training_dataset_x, TEXT_SIZE, padding='post')
test_dataset_x = pad_sequences(test_dataset_x, TEXT_SIZE, padding='post')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
model = Sequential(name='IMDB-LSTM')
model.add(Embedding(VOCAB_SIZE, 64, input_length=TEXT_SIZE, name='Embedding'))
model.add(LSTM(64, activation='tanh', name='LSTM'))
model.add(Dropout(0.5, name='Droput-1'))
model.add(Dense(128, activation='relu', name='Dense'))
model.add(Dropout(0.5, name='Droput-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
esc = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=20,
validation_split=0.2, callbacks=[esc])
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
    print(f'{model.metrics_names[i]} --> {eval_result[i]}')
    