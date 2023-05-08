# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch.nn.utils.prune as prune


def model(device, beta, spike_grad):
    net = nn.Sequential(
        nn.Linear(28*28, 100),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(100, 50),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(50, 25),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(25, 10),
        snn.Leaky(beta=beta, spike_grad=spike_grad,
                  init_hidden=True, output=True),
    ).to(device)
    return net


def forward_pass(net, num_steps, data, batch_size):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data.view(batch_size, -1))
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


def batch_accuracy(train_loader, net, num_steps, device, batch_size):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data, batch_size)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total


def training(net, optimizer, loss_fn, device, train_loader, test_loader, num_steps, batch_size):
    num_epochs = 1
    loss_hist = []
    acc_hist = []
    test_acc_hist = []
    counter = 0

    # Outer training loop
    for epoch in range(num_epochs):

        # Training loop
        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, _ = forward_pass(net, num_steps, data, batch_size)

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            acc = SF.accuracy_rate(spk_rec, targets)
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
            acc_hist.append(acc.item())
            # Test set
            if counter % 50 == 0:
                with torch.no_grad():
                    net.eval()

                    # Test set forward pass
                    test_acc = batch_accuracy(
                        test_loader, net, num_steps, device, batch_size)
                    print(
                        f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())

            counter += 1
    plots(loss_hist, 'Train Loss', 'Iteration', 'Loss')
    plots(test_acc_hist, 'Test Accuracy', 'Epoch', 'Accuracy')


def plots(data, title, xlabel, ylabel):
    # Plot Loss
    fig = plt.figure(facecolor="w")
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def create_dataset(data_path, batch_size):
    # Define a transform
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(
        data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(
        data_path, train=False, download=True, transform=transform)

    train_set, retrain_set = torch.utils.data.random_split(mnist_train, [
                                                           30000, 30000])
    test_set, retest_set = torch.utils.data.random_split(mnist_test, [
                                                         5000, 5000])
    # Create DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    retrain_loader = DataLoader(
        retrain_set, batch_size=batch_size, shuffle=True, drop_last=True)
    retest_loader = DataLoader(
        retest_set, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader, retrain_loader, retest_loader
