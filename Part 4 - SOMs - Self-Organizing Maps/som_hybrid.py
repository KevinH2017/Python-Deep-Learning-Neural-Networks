# SOM Hybrid

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from minisom import MiniSom
import tensorflow as tf
import os
dirname = os.path.dirname(__file__)

tf.__version__

### Part 1: Unsupervised Deep Learning ###

# Import dataset
dataset = pd.read_csv(dirname+"\\example_datasets\\Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Transform data into a given range to be used in the formula for normalization or standardization
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# Training the SOM:
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)

som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualize the results
plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plt.plot(
        w[0]+0.5, 
        w[1]+0.5, 
        markers[Y[i]],
        markeredgecolor=colors[Y[i]],
        markerfacecolor='None',
        markersize=10,
        markeredgewidth=2)
plt.show()

# Finding the frauds
mappings = som.win_map(X)

# Creates non-empty arrays from mappings to prevent ValueError
# results = [arr for arr in (mappings[(8,1)], mappings[(6,8)]) if arr]
frauds = np.concatenate([arr for arr in (mappings[(8,1)], mappings[(6,8)]) if len(arr) > 0], axis=0)
frauds = sc.inverse_transform(frauds)


### Part 2: Supervised Deep Learning ###

# Creating matrix of features
customers = dataset.iloc[:, 1:].values

# Creating dependent variables
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Initializes ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=4)

# Predicting the probabilities of fraud
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
print(y_pred)