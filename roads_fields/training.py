import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np


from roads_fields.dataloading import get_dataloaders
from roads_fields.utils import OUTPUT
from roads_fields.models import get_model

import time


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    correct_preds = (torch.max(model(xb), 1)[1] == yb).sum().item()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), correct_preds, len(xb)


def train_model(
    model, train_loader, valid_loader, epochs, criterion, optimizer, output_path
):
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters = {params_count}")

    start_time = time.time()

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(epochs):
        model.train()
        losses, corrects, nums = zip(
            *[
                loss_batch(model, criterion, X_train, y_train, optimizer)
                for X_train, y_train in train_loader
            ]
        )
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        train_losses.append(train_loss)
        train_accuracy = np.sum(corrects) / np.sum(nums)
        train_accuracies.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(
                *[loss_batch(model, criterion, xb, yb) for xb, yb in valid_loader]
            )
        valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        valid_losses.append(valid_loss)
        valid_accuracy = np.sum(corrects) / np.sum(nums)
        valid_accuracies.append(valid_accuracy)
        print(epoch, valid_loss, valid_accuracy)

    torch.save(model.state_dict(), output_path / "model.pt")

    plt.plot(train_losses, label="training loss")
    plt.plot(valid_losses, label="validation loss")
    plt.title("Loss at the end of each epoch")
    plt.legend()
    plt.savefig(output_path / "losses.png")
    plt.close()

    plt.plot(train_accuracies, label="training accuracy")
    plt.plot(valid_accuracies, label="validation accuracy")
    plt.title("Accuracy at the end of each epoch")
    plt.legend()
    plt.savefig(output_path / "accuracies.png")
    plt.close()

    print(f"\nDuration: {time.time() - start_time:.0f} seconds")


def train(model_name, epochs, lr, batch_size):
    output_path = OUTPUT / model_name

    os.makedirs(output_path, exist_ok=True)

    train_loader, valid_loader, class_weight = get_dataloaders(batch_size)
    model = get_model(model_name)
    train_model(
        model,
        train_loader,
        valid_loader,
        epochs,
        nn.CrossEntropyLoss(weight=class_weight),
        torch.optim.Adam(model.parameters(), lr=lr),
        output_path,
    )
