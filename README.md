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

The [`utils`](/utils) directory contains all of our utility functions broken into two categories: data-related utility functions located within [`data.py`](/utils/data.py) and model-related utility functions located within [`model.py`](/utils/model.py). Our data utility functions include loading the [MNIST](https://en.wikipedia.org/wiki/MNIST_database), [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist), and [CIFAR10](https://en.wikipedia.org/wiki/CIFAR-10) datasets. Our implementations of functions related to subsetting and sampling from these datasets (in both IID and non-IID settings) are contained within this file. Our model utility functions include vectorizing a model, saving and loading a model object for future use, and getting any available GPUs.

The [`src`](/src) directory contains the core implementation of our algorithm. We define our CNNs within [`net.py`](/src/net.py), one for use with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) and [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist) datasets and one for the [CIFAR10](https://en.wikipedia.org/wiki/CIFAR-10) dataset. Instead of storing all client information within a dictionary object as we did in Programming Assignment 3, we decided to instead define a Client class that is located within [`client.py`](/src/client.py). Finally, we also created a Server class located in [`server.py`](/src/server.py) that is flexible to work in both IID and non-IID settings. It contains the core implementation of our method.