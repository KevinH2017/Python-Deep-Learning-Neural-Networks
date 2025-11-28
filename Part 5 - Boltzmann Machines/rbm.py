# Restricted Boltzmann Machine (RBM)

import numpy as np
import pandas as pd
import torch, os
dirname = os.path.dirname(__file__) + "\\example_datasets\\"

# Import the datasets
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

# Converts the data into an array with users in lines and movies in columns
def convert(data):
    '''
    Takes multiple arrays of data (users, movies, ratings) and combines them into one array.
    Users in rows, movies in columns, and ratings in cells.
    Returns list.
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

# Convert the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1            # Empty value (Not rated by user)
training_set[training_set == 1] = 0             # Not Liked
training_set[training_set == 2] = 0             # Not Liked
training_set[training_set >= 3] = 1             # Liked

test_set[test_set == 0] = -1            # Empty value (Not rated by user)
test_set[test_set == 1] = 0             # Not Liked
test_set[test_set == 2] = 0             # Not Liked
test_set[test_set >= 3] = 1             # Liked

# Create the architecture of the neural network
class RBM():
    def __init__(self, nv, nh):
        '''Initializes random weights for visible and hidden layers.'''
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        '''
        Takes input dataset to calculate the weighted input to the nodes in the hidden layer using visible units.
        Adds hidden biases to the activations.
        Converts the activations into probabilities between 0 and 1.

        Args:
            x: The visible layer
        '''
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)        # torch.bernoulli() samples the active binary valules from the distribution.

    def sample_v(self, y):
        r'''
        Takes input dataset to calculate the weighted input to the nodes in the visible layer using hidden data.
        Adds hidden biases to the activations.
        Converts the activations into probabilities between 0 and 1.

        Args:
            y: The hidden layer
        '''
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)        # torch.bernoulli() samples the active binary valules from the distribution.

    def train(self, v0, vk, ph0, phk):
        '''
        Updates RBM's weights and biases for learning.
        Using matrix multiplication and the sum of all elements from input.

        Args:
            v0: Initial visible units
            vk: Reconstructed visible units
            ph0: Probabilities of hidden units using original visible input
            phk: Probabilities of hidden units using reconstructed visible input
        '''
        self.W += torch.mm(v0.t(), ph0).t() - torch.mm(vk.t(), phk).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# Creating RBM object
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
# Runs for 10 passes through the dataset (epoch)
nb_epoch = 10
print("Training the RBM...")
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    # Iterates through each batch
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]               # Input batch
        v0 = training_set[id_user:id_user+batch_size]               # Updated input batch
        ph0,_ = rbm.sample_h(v0)                                    # Calculates probabilities of hidden units based on visible data
        # Alternates between hidden and visible units to generate samples for the model
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]                     # Fills in missing data
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)                     # Passes in processed batches for model training
        # Adds the absolute mean between the input data and the RBM's processed output to training loss
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print("Epoch: " + str(epoch) + " Loss: " + str((train_loss/s).item()))

# Testing the RBM
test_loss = 0
s = 0.
# Iterates through each batch
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
print("Test loss: " + str((test_loss/s).item()))