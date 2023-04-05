import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import AlexNet_Weights, ResNet50_Weights

from roads_fields.dataloading import WIDTH, HEIGHT
from roads_fields.utils import OUTPUT

MODEL_NAMES = ["MyCNN", "AlexNet", "ResNet50"]


class MyCNN(nn.Module):
    def __init__(self, conv_kernels, dense_dims, kernel_size=3, stride=1):
        assert (
            len(conv_kernels) > 0 and len(dense_dims) > 0
        ), "conv_kernels and dense_dims must be int lists of size > 0"
        super().__init__()
        self.convolutional_layers = []
        self.dense_layers = []

        width_dim = WIDTH
        height_dim = HEIGHT
        n_in = 3
        for conv_kernel in conv_kernels:
            layer = nn.Conv2d(n_in, conv_kernel, kernel_size=kernel_size, stride=stride)
            self.convolutional_layers.append(layer)
            n_in = conv_kernel
            width_dim = (width_dim - 2) // 2
            height_dim = (height_dim - 2) // 2

        self.post_convolutions_dim = width_dim * height_dim * conv_kernels[-1]

        n_in = self.post_convolutions_dim
        for dim in dense_dims:
            layer = nn.Linear(n_in, dim)
            self.dense_layers.append(layer)
            n_in = dim
        layer = nn.Linear(dense_dims[-1], 2)
        self.dense_layers.append(layer)

        self.layers = nn.Sequential(*(self.convolutional_layers + self.dense_layers))

    def forward(self, X):
        for layer in self.convolutional_layers:
            X = layer(X)
            X = F.relu(X)
            X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.post_convolutions_dim)

        for layer in self.dense_layers[:-1]:
            X = layer(X)
            X = F.relu(X)

        X = self.dense_layers[-1](X)

        return F.log_softmax(X, dim=1)


def get_model(model_name):
    if model_name == "MyCNN":
        return MyCNN(conv_kernels=[6, 20, 20], dense_dims=[1024, 128])
    elif model_name == "AlexNet":
        model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # We want to freeze the pre-trained weights & biases
        for param in model.parameters():
            param.requires_grad = False

        # Next we modify the fully connected layers to produce a binary output.
        model.classifier = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 2),
            nn.LogSoftmax(dim=1),
        )
        return model
    elif model_name == "ResNet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # We want to freeze the pre-trained weights & biases
        for param in model.parameters():
            param.requires_grad = False

        # Next we modify the fully connected layers to produce a binary output.
        model.fc = nn.Sequential(
            nn.Linear(2048, 2),
            nn.LogSoftmax(dim=1),
        )
        return model
    else:
        raise Exception(f"{model_name} is not a valid model name ({MODEL_NAMES})")


def load_model(model_name):
    path = OUTPUT / model_name / "model.pt"
    model = get_model(model_name)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
