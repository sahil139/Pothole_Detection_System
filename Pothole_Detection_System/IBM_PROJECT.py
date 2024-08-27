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
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, GlobalAveragePooling2D, Dense, Dropout, Flatten
global size
def MyModel():
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', activation='relu', input_shape=(size,size,1)))
        model.add(MaxPooling2D(pool_size=(1,1)))
        model.add(Conv2D(32, (6, 6), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(1,1)))
        model.add(Conv2D(64, (4, 4), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(1,1)))
        model.add(Conv2D(128, (4, 4), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(1,1)))
        model.add(Conv2D(256, (2, 2), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(1,1)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.05))
        model.add(Dense(2, activation='softmax'))
        return model
size = 100
potholeTrainImages = glob.glob("train/Pothole/*.jpg")
potholeTrainImages.extend(glob.glob("train/Pothole/*.jpeg"))
potholeTrainImages.extend(glob.glob("train/Pothole/*.png"))

plainTrainImages = glob.glob("train/Plain/*.jpg")
plainTrainImages.extend(glob.glob("train/Plain/*.jpeg"))
plainTrainImages.extend(glob.glob("train/Plain/*.png"))

train1 = [cv2.imread(img,0) for img in potholeTrainImages]
train1 = [cv2.resize((img,(size,size)) for img in train1)]
temp1 = np.asarray(train1)

train2 = [cv2.imread(img,0) for img in plainTrainImages]
train2 = [cv2.resize((img,(size,size)) for img in train2)]
temp2 = np.asarray(train2)

plainTestImages = glob.glob("train/Plain/*.jpg")
plainTestImages.extend(glob.glob("train/Plain/*.jpeg"))
plainTestImages.extend(glob.glob("train/Plain/*.png"))

test2 = [cv2.imread(img,0) for img in plainTestImages]
test2 = [cv2.resize((img,(size,size)) for img in test2)]
temp4 = np.asarray(test2)

potholeTestImages = glob.glob("train/Pothole/*.jpg")
potholeTestImages.extend(glob.glob("train/Plain/*.jpeg"))
potholeTestImages.extend(glob.glob("train/Plain/*.png"))

test1 = [cv2.imread(img,0) for img in potholeTestImages]
test1 = [cv2.resize((img,(size,size)) for img in test1)]
temp3 = np.asarray(test1)

X_train = []
X_train.extend(temp1)
X_train.extend(temp2)
X_train = np.asarray(X_train)

X_test = []
X_test.extend(temp3)
X_test.extend(temp4)
X_test = np.asarray(X_test)

y_train1 = np.ones([temp1.shape[0]],dtype = int)
y_train2 = np.zeros([temp2.shape[0]],dtype = int)
y_test1 = np.ones([temp3.shape[0]],dtype = int)
y_test2 = np.zeros([temp4.shape[0]],dtype = int)

y_train = []
y_train.extend(y_train1)
y_train.extend(y_train2)
y_train = np.asarray(y_train)

y_test = []
y_test.extend(y_test1)
y_test.extend(y_test2)
y_test = np.asarray(y_test)

random.shuffle(X_train)
random.shuffle(y_train)
np.random.shuffle(X_test)
np.random.shuffle(y_test)

X_train = X_train.reshape(X_train.shape[0], size, size, 1)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

inputShape = (size, size, 1)
model = MyModel()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20,validation_split=0.1)

metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

