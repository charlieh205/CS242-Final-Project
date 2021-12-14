'''
Model utilities
'''
import torch
import pickle


# vectorize model
def vectorize_model(model):
    parameters = [p.data for p in model.parameters()]
    return torch.nn.utils.parameters_to_vector(parameters)


# try getting GPU (for moving tensors across devices)
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


# save object
def save_obj(obj, name, dataset_name):
    file = open(f"checkpoints_{dataset_name}/{name}.obj", "wb")
    pickle.dump(obj, file)


# load object
def load_obj(name, dataset_name):
    file = open(f"checkpoints_{dataset_name}/{name}.obj", "rb")
    return pickle.load(file)
