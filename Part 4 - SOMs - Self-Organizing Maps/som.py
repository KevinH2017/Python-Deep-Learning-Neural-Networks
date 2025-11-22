# Self-Organizing Maps (SOM)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import os
dirname = os.path.dirname(__file__)

# Import dataset
dataset = pd.read_csv(dirname+"\\example_datasets\\Credit_Card_Applications.csv")

# Customer ID, A1 to A14 columns and values
X = dataset.iloc[:, :-1].values
# Class column and values
Y = dataset.iloc[:, -1].values

# Transform data into a given range to be used in the formula for normalization or standardization
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

'''
Training the SOM:

x = x dimension of the SOM
y = y dimension of the SOM
input_len = number of elements of the vectors in input
sigma = the spread of the neighborhood function
learning_rate = the initial learning rates
'''
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
# Initializes training weights
som.random_weights_init(X)
# Picks random data for training
som.train_random(data=X, num_iteration=100)

# Visualize the results
plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()
# Assigns each node a color based on if the customer got approval or not
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
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs:')
for i in frauds[:, 0]:
  print(int(i))