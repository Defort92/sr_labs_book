# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script creates different optimizers


# Import of modules
import torch

from constants import ADAM_INITIALIZATION_MESSAGE, SGD_INITIALIZATION_MESSAGE


def AdamOptimizer(parameters, lr, weight_decay, **kwargs):
    # Function to create Adam optimizer

    print(ADAM_INITIALIZATION_MESSAGE)

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay);

def SGDOptimizer(parameters, lr, weight_decay, **kwargs):
    # Function to create SGD optimizer

    print(SGD_INITIALIZATION_MESSAGE)

    return torch.optim.SGD(parameters, lr = lr, momentum = 0.9, weight_decay=weight_decay)
