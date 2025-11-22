# Recurrent Neural Network (RNN)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os
dirname = os.path.dirname(__file__)

# Import training dataset
dataset_train = pd.read_csv(dirname+"\\example_datasets\\Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Transform data into a given range to be used in the formula for normalization or standardization
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating data structure with 60 timesteps and 1 output
x_train = list()
y_train = list()
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping ensures that the data is aligned with the expected input format of the neural network
# This is achieved by taking in each output from each timestep over the course of the entire dataset
# EX: (60, 1258, 1) --> 60 timesteps of the 1258 rows and then process 1 output from each timestep over the course of the reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Initializes the neural network
regressor = Sequential()
print("RNN Initialized")

'''
First LSTM layer

units is the number of neurons in the first layer
return_sequences returns the last output in the output sequence, letting you add another LSTM layer
input_shape is the reshaped training data created using np.reshape() on x_train
'''
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# Dropout Layer
# Neurons to be ignored / dropped for each step during training to make it more accurate
regressor.add(Dropout(0.2))
print("LSTM Layer 1 Added")

# Second LSTM Layer
# The input_shape is already set in the first layer, so it is removed for future layers
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
print("LSTM Layer 2 Added")

# Third LSTM Layer:
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
print("LSTM Layer 3 Added")

# Fourth LSTM Layer
# The return_sequences is removed for the last layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
print("LSTM Layer 4 Added")

# Output Layer
regressor.add(Dense(units=1))
print("Output Layer Added")

# Compile the Neural Network
# Mean Squared Error (MSE) helps analyze the accuracy of a model and gets the average of the differences
# between the predicted values and the actual values in the dataset
regressor.compile(optimizer='adam', loss='mean_squared_error')
print("RNN compiling completed succesfully!")

# Fitting and Training the Neural Network
regressor.fit(x_train, y_train, epochs=100, batch_size=32)
print("RNN training completed succesfully!")

# Import test dataset
dataset_test = pd.read_csv(dirname+"\\example_datasets\\Google_Stock_Price_Test.csv")
# Real stock price of 2017
real_stock_price = dataset_test.iloc[:, 1:2].values

# Dataframe that predicts the stock price of 2017, using the 'Open' column
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# Create and reshape test dataset
x_test = list()
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualize the results
plt.plot(real_stock_price, color='blue', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='green', label='Predicted Google Stock Price')
plt.title("Google Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()