'''
Model utilities
'''
import torch


# vectorize model
def vectorize_model(model):
    parameters = [p.data for p in model.parameters()]
    return torch.nn.utils.parameters_to_vector(parameters)


# try getting GPU (for moving tensors across devices)
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")
