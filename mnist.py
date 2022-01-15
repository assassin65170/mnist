import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets,models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
mnist = datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(train_images[2])
plt.show()
print(train_labels[2])

train_images = train_images /255
test_images = test_images /255
print(train_images[2])

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

pred = model.predict([test_images], verbose=0)
print(np.argmax(pred[1823]))
plt.imshow(test_images[1823])
plt.show()

