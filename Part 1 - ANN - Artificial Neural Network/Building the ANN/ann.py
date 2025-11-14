import tensorflow as tf
# keras is now part of tensorflow, so we can import from tensorflow directly
from keras.layers import Dense
from keras.models import Sequential

tf.__version__

# Initializing the ANN
ann = Sequential()

'''
Adding the input layer and the first hidden layer
Dense(units, activation) - 
    units: number of neurons in the layer
    activation: this is the mathematical equation used to get the output of a Node

relu function example in pseudo code:
    if input > 0:
        return input
    else:
        return 0
'''
ann.add(Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(Dense(units=6, activation='relu'))

'''
Adding the output layer
One unit because it's a binary classification problem (0 or 1)
Sigmoid activation function because it will give a value between 0 and 1

sigmoid function example in pseudo code:
    import numpy as np 
    def sig(x):
        return 1/(1 + np.exp(-x))
'''
ann.add(Dense(units=1, activation='sigmoid'))
