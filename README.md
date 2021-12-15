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

The [`utils`](/utils) directory contains all of our utility functions broken into two categories: data-related utility functions located within [`data.py`](/utils/data.py) and model-related utility functions located within [`modes.py`](/utils/model.py). Our data utility functions include loading the [MNIST](https://en.wikipedia.org/wiki/MNIST_database), [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist), and [CIFAR10](https://en.wikipedia.org/wiki/CIFAR-10) datasets. Our implementations of functions related to subsetting and sampling from these datasets (in both IID and non-IID settings) are contained within this file. Our model utility functions include vectorizing a model, saving and loading a model object for future use, and getting any available GPUs.