import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2, glob
import random
from keras.models import Sequential, Model
from keras.metrics import Precision
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, GlobalAveragePooling2D, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

def MyModel(input_shape):
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (6, 6), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Conv2D(64, (4, 4), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (4, 4), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256, (2, 2), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    return model

size = 100

potholeTrainImages = glob.glob("Pothole1/*.jpg")
potholeTrainImages.extend(glob.glob("Pothole1/*.jpeg"))
potholeTrainImages.extend(glob.glob("Pothole1/*.png"))

plainTrainImages = glob.glob("Plain1/*.jpg")
plainTrainImages.extend(glob.glob("Plain1/*.jpeg"))
plainTrainImages.extend(glob.glob("Plain1/*.png"))

train1 = [cv2.imread(img, 0) for img in potholeTrainImages]
train1 = [cv2.resize(img, (size, size)) for img in train1]
temp1 = np.asarray(train1)

train2 = [cv2.imread(img, 0) for img in plainTrainImages]
train2 = [cv2.resize(img, (size, size)) for img in train2]
temp2 = np.asarray(train2)

X_train = np.concatenate((temp1, temp2), axis=0)
y_train = np.concatenate((np.ones(temp1.shape[0]), np.zeros(temp2.shape[0])), axis=0)

X_train = X_train / 255.0

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

X_train = X_train.reshape(X_train.shape[0], size, size, 1)
X_val = X_val.reshape(X_val.shape[0], size, size, 1)

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

print("Train shape X:", X_train.shape)
print("Train shape y:", y_train.shape)
inputShape = (size, size, 1)


model = MyModel(inputShape)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))


