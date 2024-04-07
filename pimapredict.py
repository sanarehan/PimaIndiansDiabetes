import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
diabetes = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

# Split features and target variable
X = diabetes.iloc[:, :-1]
y = diabetes.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the input shape
input_shape = X_train.shape[1]

# Build the model
model = keras.Sequential([
    keras.layers.Dense(units=64, activation="relu", input_shape=(input_shape,)),
    keras.layers.Dense(units=32, activation="relu"),
    keras.layers.Dense(units=20, activation="relu"),
    keras.layers.Dense(units=1, activation="tanh")
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc)
