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

Next, we'll look at how we perform on non-IID with no skew. This is done in much the same fashion as in the [IID setting](#iid-setting), but our number of clients will be a tuple indicating the number of clients for each class. We will also define our classes for the server.
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

### Non-IID setting - some skew

To test on non-IID with some skew, we'll instantiate our server almost identically to that of [non-IID with no skew](#non-iid-setting---no-skew). The only difference is the number of clients that we assign to each class.
```python
server = Server(
    net,
    trainset,
    testset,
    num_clients = (20, 10, 10),
    classes = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
)
server.train(rounds = 50, local_epochs = 5, weighted = True)
```

### Non-IID setting - extreme skew

Finally, to test on extreme skew, we'll define our server identically to that of [non-IID with some skew](#non-iid-setting---some-skew) but assign an even higher number of clients to the first class.
```python
server = Server(
    net,
    trainset,
    testset,
    num_clients = (40, 10, 10),
    classes = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
)
server.train(rounds = 50, local_epochs = 5, weighted = True)
```

### Saving results

In order to avoid losing any results that may occur when one is away from their computer, we defined a function that allows us to save the server object to disk after it is done training. It can then be loaded later. These functions are located in our utility file [`model.py`](/utils/model.py). To demonstrate, we'll say that we are testing on the MNIST dataset under IID settings using FedWeighted. We can save the final results with the ``save_obj`` function. The first argument is the server object to be save, the second specifies the setting and algorithm used, and the third is the dataset.
```python
save_obj(server, "IID_weighted", "MNIST")
```
Executing this will create a new directory ``checkpoints_MNIST/``, within which there will be a file ``IID_weighted.obj``. To access this file, we can simply use the ``load_obj`` function with the same last two arguments.
```python
server = load_obj("IID_weighted", "MNIST")
```
This allows us to load in results at later times for more experimentation and analysis.