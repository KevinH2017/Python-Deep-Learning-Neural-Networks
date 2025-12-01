# AutoEncoders

import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
dirname = os.path.dirname(__file__) + "\\example_datasets\\"

# Import the dataset
movies = pd.read_csv(dirname+"ml-1m\\movies.dat", sep="::", header=None, engine='python', encoding='latin-1')
users = pd.read_csv(dirname+"ml-1m\\users.dat", sep="::", header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(dirname+"ml-1m\\ratings.dat", sep="::", header=None, engine='python', encoding='latin-1')

# Preparing the training and test sets
training_set = pd.read_csv(dirname+"ml-100k\\u1.base", delimiter='\t')
training_set = np.array(training_set, dtype="int")                                      # Puts data into an array of lists of integers
test_set = pd.read_csv(dirname+"ml-100k\\u1.test", delimiter='\t')
test_set = np.array(test_set, dtype="int")                                              # Puts data into an array of lists of integers

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))                         # Gets maximum value of all rows of the first column
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))                        # Gets maximum value of all rows of the second column

# Converts the data into an array with users in lines, movies in columns, ratings in cells
def convert(data):
    '''
    Takes multiple arrays of data (users, movies, ratings) and combines them into one array.
    Users in rows, movies in columns, and ratings in cells.

    :return: List of converted arrays
    '''
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the neural network
class SAE(nn.Module):
    def __init__(self, ):
        '''Stacked autoencoder to initialize neural network layers and Sigmoid function.'''
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)         # First Layer - Input 20 neurons
        self.fc2 = nn.Linear(20, 10)                # Second Layer - Hidden layer 10 neurons
        self.fc3 = nn.Linear(10, 20)                # Third Layer - Hidden layer 20 neurons
        self.fc4 = nn.Linear(20, nb_movies)         # Fourth Layer - Reconstructs to same size as original input
        self.activation = nn.Sigmoid()              # Sigmoid activation function

    def forward(self, x):
        '''
        Forward propagation. Input goes through encoding layers (fc1, fc2) to be compressed.
        Then goes through decoding layers (fc3, fc4) to be reconstructed to the same size as the original input.
        The Sigmoid function is applied after each layer except for the final output layer.

        :return: :list: Returns final output layer.
        '''
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# Create instance of SAE (Stacked Autoencoder) class
sae = SAE()
# Mean Squared Error (MSE) Loss to measure differences between input and reconstructed output
criterion = nn.MSELoss()
# Updates weights during training using learning rate and weight decay to avoid overfitting and inaccurate results
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE (Stacked Autoencoder)
# Runs for 200 passes through the dataset (epoch)
nb_epoch = 200
print("Training the SAE...")
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    # Iterates through each batch
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)    # Tensor object of user movie ratings
        target = input.clone()                                  # Clones user movie ratings object
        # Only train with users that rated at least one movie
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.requires_grad = False
            output[target == 0] = 0                                     # Fills in empty vectors with 0 to avoid errors
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()                                             # Backpropagation
            train_loss += np.sqrt(loss.item() * mean_corrector)         # Gets values from 0-dim tensor to numbers to avoid IndexError
            s += 1.
            optimizer.step()                                            # Updates model's weights
    print("Epoch: " + str(epoch) + " Loss: " + str((train_loss/s).item()))

# Testing the SAE (Stacked Autoencoder)
test_loss = 0
s = 0.
print("Testing the SAE...")
# Iterates through each batch
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)    # Tensor object of users' movie ratings
    target = input.clone()                                  # Clones input variable
    # Only train with users that rated at least one movie
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.requires_grad = False
        output[target == 0] = 0                                     # Fills in empty vectors with 0 to avoid errors
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        loss.backward()                                             # Backpropagation
        test_loss += np.sqrt(loss.item() * mean_corrector)          # Gets values from 0-dim tensor to numbers to avoid IndexError
        s += 1.
        optimizer.step()                                            # Updates model's weights
print("Test loss: " + str((test_loss/s).item()))
