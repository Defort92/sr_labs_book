# Load and save pth files


# Import of modules
import os

import torch

from constants import CHECKPOINT_FILENAME_TEMPLATE


def saveParameters(model, optimizer, scheduler, num_epoch, path):
    # Save parameters
    
    checkpoint = {}
    checkpoint['model']     = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['scheduler'] = scheduler[0].state_dict()
    checkpoint['num_epoch'] = num_epoch
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    filename = CHECKPOINT_FILENAME_TEMPLATE.format(epoch=num_epoch)
    torch.save(checkpoint, os.path.join(path, filename))
    
def loadParameters(model, optimizer, scheduler, path):
    # Load parameters

    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler[0].load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['num_epoch']
