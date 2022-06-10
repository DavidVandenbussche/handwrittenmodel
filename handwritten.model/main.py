# Open paint, resize for 28*28, save image in digits/ as digit1.png and run

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('handwritten.model')

img = cv2.imread(f"digits/digit1.png")[:,:,0] # not interested in colors
img = np.invert(np.array([img])) # by default it's white on black and not black on white, array to use it in the network
prediction = model.predict(img)
print(prediction)
print(f"this digit is probably a {np.argmax(prediction)}")
