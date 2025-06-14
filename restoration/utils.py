import numpy as np

import json

from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision import transforms
import torchvision

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

with open('config/config.json') as json_file:
    config = json.load(json_file)

ROOT_DATASET = config['ROOT_DATASET']


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Example usage
set_seed(42)

def to_complex(x):
    x_ = torch.moveaxis(x, 1, -1).contiguous()
    return torch.view_as_complex(x_)


def to_image(x, clamp=True, rescale=False):
    if x.shape[1] == 2:
        x_complex = to_complex(x).contiguous()
        out = torch.abs(x_complex)
    elif x.shape[1] == 1:
        out = x[:, 0, ...]  # keep batch dim
        out = torch.nan_to_num(out)
        if clamp:
            out = torch.clamp(out, 0, 1)
    else:
        out = torch.moveaxis(x, 1, -1).contiguous()
        out = torch.nan_to_num(out)
        if clamp:
            out = torch.clamp(out, 0, 1)
        if rescale:
            out = out - out.min()
            out = out/out.max()
    return out


def get_data(dataset_name='set3c', batch_size=1, img_size=None, n_channels=None, crop=False, padding=None):

    # Set the global random seed from pytorch to ensure reproducibility of the example.
    torch.manual_seed(0)

    test_transform = []

    if img_size is not None:
        # test_transform.append(transforms.CenterCrop(img_size))
        if crop:
            test_transform.append(transforms.CenterCrop(img_size))
        else:
            test_transform.append(transforms.Resize(img_size))
    if padding is not None:
        test_transform.append(transforms.Pad(padding))
    if n_channels is not None and n_channels == 1:
        test_transform.append(transforms.Grayscale(num_output_channels=1))
    test_transform.append(transforms.ToTensor())
    test_transform = transforms.Compose(test_transform)

    test_dataset = torchvision.datasets.ImageFolder(root=ROOT_DATASET+'/'+dataset_name, transform=test_transform)

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=4, shuffle=False
    )

    return test_dataloader
