#Import der Libraries
import tensorflow as tf
import numpy as np

#in diesem Abschnitt werden die Daten (Bilder und Labels) aus der Datenbank "Mnist Fashion" eingeladen
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#in diesem Abschnitt werden die Daten so umgeformt, dass das neuronale Netz mit den Daten arbeiten kann
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

#in diesem Abschnitt wird das neuronale Netz "gebaut"
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#in diesem Abschnitt wird das neuronale Netz trainiert
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images,training_labels, epochs=4)

#in diesem Abschnitt wird das neuronale Netz mit den Testbildern des Datensets getestet, welche das neuronale Netz noch nie gesehen hat
classes = model.predict(test_images)
predicted_classes = np.argmax(classes, axis=1)
print(classes[0])
print(test_labels[0])

#in diesem Abschnitt wird das Bild betrachtet, welches vorher zum Testen genutzt wurde

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(test_images[0], cmap='Greys_r')