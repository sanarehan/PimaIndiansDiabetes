import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

diabetes = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

X = diabetes.iloc[:, :-1]
y = diabetes.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

input_shape = X_train.shape[1]

model = keras.Sequential([
    keras.layers.Dense(units=64, activation="relu", input_shape=(input_shape,)),
    keras.layers.Dense(units=32, activation="relu"),
    keras.layers.Dense(units=20, activation="relu"),
    keras.layers.Dense(units=1, activation="tanh")
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc)
