# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:22:21 2021

@author: alisa
"""
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

from tensorflow.keras.datasets import cifar100
(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = cifar100.load_data()
class_names = [
'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
'worm'
]
print('Shape of training_dataset_x: {}'.format(training_dataset_x.shape))
print('Shape of training_dataset_y: {}'.format(training_dataset_y.shape))
print('Shape of test_dataset_x: {}'.format(test_dataset_x.shape))
print('Shape of test_dataset_y: {}'.format(test_dataset_y.shape))
import matplotlib.pyplot as plt
for i in range(10):
    print(class_names[training_dataset_y[i][0]])
    plt.imshow(training_dataset_x[i])
    plt.pause(1)
image_index = class_names.index('sweet_pepper')
for index, dataset in enumerate(training_dataset_x[(training_dataset_y == image_index).reshape(-1)]):
    plt.imshow(dataset)
    plt.pause(1)
    if index == 10:
        break
training_set_x = training_dataset_x.astype('float32')
test_set_x = test_dataset_x.astype('float32')
training_dataset_x = training_dataset_x / 255
test_dataset_x = test_dataset_x / 255
from tensorflow.keras.utils import to_categorical
training_dataset_y = to_categorical(training_dataset_y)
test_dataset_y = to_categorical(test_dataset_y)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu',
name='Convolution-1'))
model.add(MaxPooling2D(name='MaxPooling-1'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='Convolution-2'))
model.add(MaxPooling2D(name='MaxPooling-2'))
model.add(Flatten(name='Flatten'))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(100, activation='softmax', name='Output'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, epochs=12, batch_size=512,
validation_split=0.2)
import matplotlib.pyplot as plt
#plt.title('Epoch-Accuracy Graph')
#plt.xlabel = 'Epochs'
#plt.ylabel = 'Loss'
#plt.plot(range(1, len(hist.epoch) + 1), hist.history['acc'])
#plt.plot(range(1, len(hist.epoch) + 1), hist.history['val_acc'])
#plt.legend(['acc', 'val_acc'])
loss, accuracy = model.evaluate(test_dataset_x, test_dataset_y)
print('loss = {}, accuracy = {}'.format(loss, accuracy))
from matplotlib.pyplot import imread
import glob
for path in glob.glob('tır.png'):
    img = imread(path)
    img = img.reshape(1, 32, 32, 3)
    predict_result = model.predict(img)
    number = predict_result.argmax(axis=1)
    print('{} -> {}'.format(path, class_names[number[0]]))