# Density-Based Client Weighting
CS242 Final Project for Team 7  
Team members:
- Charlie Harrington, <charlesharrington@g.harvard.edu>  
- Cole French, <cfrench@college.harvard.edu>  
- Matthew Nazari, <matthewnazari@college.harvard.edu>  
- Michael Cheng, <michaelcheng@college.harvard.edu>

## Organization

### Directory structure

Our directory structure is as follows.

    CS242-Final-Project/
    ├── src
    │   ├── client.py
    │   ├── net.py
    │   └── server.py
    ├── utils
    │   ├── data.py
    │   └── model.py
    ├── .gitignore
    ├── README.md
    ├── main.ipynb
    ├── plot.ipynb
    ├── requirements.txt
    └── test.ipynb

### Descriptions

The [`utils`](/utils) directory contains all of our utility functions broken into two categories: data-related utility functions located within [`data.py`](/utils/data.py) and model-related utility functions located within [`model.py`](/utils/model.py). Our data utility functions include loading the [MNIST](https://en.wikipedia.org/wiki/MNIST_database), [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist), and [CIFAR10](https://en.wikipedia.org/wiki/CIFAR-10) datasets. Our implementations of functions related to subsetting and sampling from these datasets (in both IID and non-IID settings) are contained within this file. Our model utility functions include vectorizing a model, saving and loading a model object for future use, and getting any available GPUs.

The [`src`](/src) directory contains the core implementation of our algorithm. We define our CNNs within [`net.py`](/src/net.py), one for use with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) and [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist) datasets and one for the [CIFAR10](https://en.wikipedia.org/wiki/CIFAR-10) dataset. Instead of storing all client information within a dictionary object as we did in Programming Assignment 3, we decided to instead define a Client class that is located within [`client.py`](/src/client.py). Finally, we also created a Server class located in [`server.py`](/src/server.py) that is flexible to work in both IID and non-IID settings. It contains the core implementation of our method.

## How to use

This repository is designed to allow for quick and easy experimentation. Its modularity allows users to use and test all datasets in various IID settings. To demonstrate, we will walk through an example in which we are interested in testing our FedWeighted algorithm on the MNIST dataset. First, we need to import the relevant functions and classes.
```python
from utils.data import *
from utils.model import *
from src.client import Client
from src.serer import Server
from src.net import MNISTNet
```
Next, we'll load our training and test datasets and instantiate our CNN.
```python
trainset, testset = load_MNIST()
net = MNISTNet()
```
We're now ready to run our experiments.

### IID setting

We first want to see the baseline performance, so we'll instantiate a Server object with our net, training set, test set, and number of clients (60 for this example). To train, we simply call the server's train method and specify the number of rounds and local epochs that we want (50 and 5 for this example). **Note: by specifying ``weighted = True``, we indicate that we are using the FedWeighted algorithm. The default of ``weighted = False`` uses FedAvg.**
```python
server = Server(
    net,
    trainset,
    testset,
    num_clients = 60
)
server.train(rounds = 50, local_epochs = 5, weighted = True)
```

### Non-IID setting - no skew

Next, we'll look at how we perform on non-IID with no skew. This is done in much the same fashion as in the [IID setting](###iid-setting), but our number of clients will be a tuple indicating the number of clients for each class. We will also define our classes for the server.
```python
server = Server(
    net,
    trainset,
    testset,
    num_clients = (20, 20, 20),
    classes = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
)
server.train(rounds = 50, local_epochs = 5, weighted = True)
```