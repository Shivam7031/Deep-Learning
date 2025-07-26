# mnist_classification.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt
from streamlit.runtime.scriptrunner import get_script_run_ctx
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

X_test.shape

y_train

import matplotlib.pyplot as plt
plt.imshow(X_train[2])

X_train = X_train/255
X_test = X_test/255

X_train[0]

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=25,validation_split=0.2)

y_probe =model.predict(X_test)

y_pred =y_probe.argmax(axis=1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

plt.plot(history.history['loss'],label='train loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],label='train accuracy')
plt.plot(history.history['val_accuracy'],label='val accuracy')
plt.legend()
plt.show()

plt.imshow(X_test[2])

model.predict(X_test[2].reshape(1,28,28)).argmax(axis=1)

plt.imshow(X_test[3])

model.predict(X_test[3].reshape(1,28,28)).argmax(axis=1)

plt.imshow(X_test[401])

model.predict(X_test[401].reshape(1,28,28)).argmax(axis=1)

# Save the model to a file using Keras's save method
model.save('Web Dev/mnist_model.h5')
