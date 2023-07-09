import numpy as np
import pandas as pd
import scipy.io
import math
import os
import ntpath
import sys
import logging
import time
import sys
from sklearn.metrics import mean_squared_error
from importlib import reload
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras_tuner import BayesianOptimization
import tensorflow_addons as tfa
from keras import backend as K
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import load_model



tf.random.set_seed(314)
np.random.seed(314)
column_labels = ['Voltage', 'Current', 'Temperature', 'SOC']
df = pd.read_csv('McMaster_p25.csv', header=None, names=column_labels)


# Define the model-building function
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
                   activation='relu',
                   input_shape=(time_steps, X_reshaped.shape[2])))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss=Huber())

    return model

# Set up the Bayesian Optimization tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='results/',
    project_name='lstm_tuner'
)

# Define the callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

# Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]


column_labels = ['Voltage', 'Current', 'Temperature', 'SOC']
df = pd.read_csv('McMaster_p25_cleaned.csv', header=None, names=column_labels)
X = df[['Voltage', 'Current', 'Temperature']]
y = df['SOC']

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the input data to match LSTM requirements (samples, time_steps, features)
time_steps = 1
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], time_steps, X_scaled.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Define the path to the saved model
model_path = 'C:/Users/Harpreet Singh/Documents/Machine Learning/Thesis/best_model.h5'

# Load the saved model
best_model = load_model(model_path)

# Make predictions using the loaded model
y_pred = best_model.predict(X_test)

# Create a plot of actual vs predicted SOC
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c='red', linestyle='--')
plt.xlabel('Actual SOC')
plt.ylabel('Predicted SOC')
plt.title('Actual vs Predicted SOC')
plt.show()

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)