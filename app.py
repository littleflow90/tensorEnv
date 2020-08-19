#Beispiel einer Linearen Regression

#Import der Libraries
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

#Trainingsdaten
xs = [1, 2, 3, 4]
ys = [2, 4, 6, 8]

model.fit(xs, ys, epochs=1000)

#Predict
print(model.predict([7]))