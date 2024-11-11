# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:50:18 2023

@author: mpseb
"""
# Multilayer Perceptron (MLP) model of the MNIST dataset
# hese MLP models are also referred to as either deep feedforward networks
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt # plotting library
#%matplotlib inline


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam,RMSprop
from keras import  backend as K

# import dataset
# MNIST is a collection of handwritten digits ranging from the number 0 to 9.
# It has a training set of 60,000 images, and 10,000 test images that are 
# classified into corresponding categories or labels

from keras.datasets import mnist


# load dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# count the number of unique train labels
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))

# count the number of unique test labels
unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)))

# sample 25 mnist digits from train dataset
indexes = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]


# plot the 25 mnist digits
plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
plt.show()
#plt.savefig("mnist-samples.png")
plt.close('all')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.utils import plot_model

# The data must be in the correct shape and format.
# After loading the MNIST dataset, the number of labels is computed
# compute the number of labels
num_labels = len(np.unique(y_train))

# The labels are in digits format, 0 to 9.
# This sparse scalar representation of labels is not suitable for the 
# neural network prediction layer that outputs probabilities per class.
# A more suitable format is called a one-hot vector, a 10-dim vector with 
# all elements 0, except for the index of the digit class.
# So, convert to one-hot vector format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""
In deep learning, data is stored in tensors. The term tensor applies to a 
scalar (0D tensor), vector (1D tensor), matrix (2D tensor), and a 
multi-dimensional tensor.  

We now  compute the image dimensions, 
input_size of the first Dense layer and scales each pixel value from 
0 to 255 to range from 0.0 to 1.0. The output of the NN  is also normalized. 

After training, there is an option to put everything back to the integer 
pixel values by multiplying the output tensor by 255. 

The proposed model is based on MLP layers. Therefore, the input is expected 
to be a 1D tensor. So, x_train and x_test are reshaped to [60000, 28 28] 
and [10000, 28 28], respectively.
"""
# image dimensions (assumed square)
image_size = x_train.shape[1]
input_size = image_size * image_size
input_size

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# Setting NN parameters
# network parameters
batch_size = 128
hidden_units = 256
dropout = 0.45

# The MLP model architecture 
"""
The proposed model is made of 3 MLP layers. In Keras, an MLP layer is 
referred to as Dense, which stands for the densely connected layer.

Both the first and second MLP layers are identical in nature with 256 
units each, followed by relu activation and dropout. 256 is more appropriate 
as 128, 512 and 1,024 units have lower performance metrics. 
At 128 units, the network converges quickly, but has a lower test accuracy. 
The added number units for 512 or 1,024 does not increase the test accuracy 
significantly.

"""
# The main data structure in Keras is the Sequential class, which allows 
# the creation of a basic neural network.
from keras.models import Sequential

model = Sequential()

# Our model is a 3-layer MLP with ReLU and dropout after each layer
# In Keras, we can add the required types of layers through the add() method.
# model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))

# MNIST digit classification is inherently a non-linear process. 
# Inserting a relu activation between Dense layers will enable MLPs to 
# model non-linear mappings.
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Keras library provides us summary() method to check the model description
model.summary()
"""
The model requires a total of 269,322 parameters. This is substantial 
considering that we have a simple task of classifying MNIST digits. 
Thus we can see that MLPs are not parameter efficient.

The total number of parameters required is computed as follows:

From input to Dense layer: 784 × 256 + 256 = 200,960.

From first Dense to second Dense: 256 × 256 + 256 = 65,792.

From second Dense to the output layer: 10 × 256 + 10 = 2,570.

Total = 200,690 + 65,972 + 2,570 = 269,322.
"""
# Another way of verifying the network is by calling the plot_model() method 
plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

# Implementation of MLP model in Keras comprises of three steps:-

# Compiling the model with the compile() method.

# Training the model with fit() method.

# Evaluating the model performance with evaluate() method.

"""How far the predicted tensor is from the one-hot ground truth 
vector is called loss.

Here, we use categorical_crossentropy as the loss function. 
It is the negative of the sum of the product of the target and 
the logarithm of the prediction.

For classification by category, categorical_crossentropy or 
mean_squared_error is a good choice after the softmax activation 
layer. The binary_crossentropy loss function is normally used after 
the sigmoid activation layer while mean_squared_error is an option 
for tanh output.

With optimization, the objective is to minimize the loss function. 
Adam is used here as it has the highest test accuracy.
"""
# The default metric in Keras is loss.
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))

# The l2 weight regularizer with fraction=0.001 is implemented as
from keras.regularizers import l2
model.add(Dense(hidden_units,
                kernel_regularizer=l2(0.001),
                input_dim=input_size))

