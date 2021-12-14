'''
Data utilities
'''
import torch
from torchvision import datasets, transforms


# dataset loaders
def load_CIFAR10():
    return (
        datasets.CIFAR10(root='data/CIFAR10', transform=transforms.ToTensor(), train=True, download=True),
        datasets.CIFAR10(root='data/CIFAR10', transform=transforms.ToTensor(), train=False, download=True)
    )


def load_Fashion_MNIST():
    return (
        datasets.FashionMNIST(root='data', transform=transforms.ToTensor(), train=True, download=True),
        datasets.FashionMNIST(root='data', transform=transforms.ToTensor(), train=False, download=True)
    )


def load_MNIST():
    return (
        datasets.MNIST(root='data', transform=transforms.ToTensor(), train=True, download=True),
        datasets.MNIST(root='data', transform=transforms.ToTensor(), train=False, download=True)
    )


# get indices of certain class
def idxs_of_classes(dataset, classes):
    if isinstance(classes, int):
        classes = (classes,)
    idxs_of_class = lambda c: (torch.as_tensor(dataset.targets) == c).nonzero().squeeze()
    return torch.cat([idxs_of_class(c) for c in classes], dim=0)


# create subset of dataset
def dataset_split(dataset, idxs):
    return torch.utils.data.Subset(dataset, idxs)


# sample from a dataset
def sample_dataset(dataset, pct=None, n=None, classes=None):
    # indices of datapoints that should be sampled from (e.g. by class)
    if not classes:
        idxs = torch.arange(len(dataset))
    else:
        idxs = idxs_of_classes(dataset, classes)
    # determine number of datapoints to sample
    if not n and pct:
        n = int(pct * len(idxs))
    else:
        assert n is not None, "pct or n must be provided"
    # sample indices
    idxs = idxs.repeat(n // len(idxs) + 1)
    idxs_shuffled = idxs[torch.randperm(len(idxs))]
    idxs_sampled = idxs_shuffled[:n]
    # return dataset split
    return dataset_split(dataset, idxs_sampled)


# create dataloader
def dataloader(dataset, batch_size=100):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
