from collections import OrderedDict

import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from roads_fields.utils import TRAIN_SET

WIDTH = 300
HEIGHT = 200
IMAGE_SIZE = (HEIGHT, WIDTH)

# Recommended ImageNet normalization params
IMAGENET_NORMALIZATION = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

TEST_SIZE = 0.25

CLASSES = ["fields", "roads"]


def get_transforms(train=False):
    transformations = OrderedDict()

    # In training mode we Augment our data to prevent our model from overfitting and make it more robust
    if train:
        transformations["rand_augment"] = transforms.RandAugment()
        transformations["flip"] = transforms.RandomHorizontalFlip()

    transformations["resize"] = transforms.Resize(IMAGE_SIZE)
    transformations["to_tensor"] = transforms.ToTensor()
    transformations["normalize"] = transforms.Normalize(*IMAGENET_NORMALIZATION)

    return transforms.Compose(list(transformations.values()))


def get_dataloaders(batch_size=10):
    dataset = datasets.ImageFolder(TRAIN_SET, transform=get_transforms(train=True))

    # We need to compute class weights to take it into account in the loss as classes are imbalanced
    class_weight = torch.Tensor(
        compute_class_weight("balanced", classes=[0, 1], y=dataset.targets)
    )

    num_train = len(dataset)
    indices = list(range(num_train))
    train_idx, valid_idx = train_test_split(indices, test_size=TEST_SIZE)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, valid_loader, class_weight
