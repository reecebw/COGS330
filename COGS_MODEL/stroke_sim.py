# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch.nn.utils.prune as prune


def plot_weight_heatmap(net, subtitle):
    layers = [module for module in net.modules(
    ) if isinstance(module, nn.Linear)]

    fig, axes = plt.subplots(nrows=1, ncols=len(layers), figsize=(15, 5))
    fig.suptitle(subtitle)

    for idx, layer in enumerate(layers):
        weight_matrix = layer.weight.detach().cpu().numpy().transpose()
        im = axes[idx].imshow(weight_matrix, cmap='coolwarm', aspect='auto')
        axes[idx].set_title(f'Layer {idx + 1}')
        fig.colorbar(
            im, ax=axes[idx], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


class MaskedAdam(torch.optim.Adam):
    def __init__(self, params, masks, lr, **kwargs):
        super(MaskedAdam, self).__init__(params, lr, **kwargs)
        self.masks = masks

    def step(self, closure=None):
        for group, mask_group in zip(self.param_groups, self.masks):
            for p, m in zip(group['params'], mask_group):
                if p.grad is None or m is None:
                    continue

                # Apply weight mask before updating the gradient
                p.grad.data.mul_(m)

        # Call the original step method to update the weights
        super(MaskedAdam, self).step(closure)


def prune_weights(net, pruning_amount=0.5):
    masks = []
    layer_masks = []
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            # Apply L1 unstructured pruning
            prune.random_unstructured(
                module, name='weight', amount=pruning_amount)

            # Store the mask for the pruned weights
            layer_masks.append(module.weight_mask.clone())

            # Make the pruning permanent
            prune.remove(module, 'weight')
        else:
            layer_masks.append(None)

        if len(layer_masks) == len(net):
            masks.append(layer_masks)
            layer_masks = []

    return masks


def test_accuracy(net, num_steps, test_loader, device, batch_size):
    with torch.no_grad():
        net.eval()

        # Test set forward pass
        test_acc_post_prune = model.batch_accuracy(
            test_loader, net, num_steps, device, batch_size)
        print(f"Test Acc: {test_acc_post_prune * 100:.2f}%\n")


if __name__ == "__main__":
    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.8
    num_steps = 50

    # dataloader arguments
    batch_size = 128
    data_path = '/data/mnist'

    dtype = torch.float
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, test_loader, retrain_loader, retest_loader = model.create_dataset(
        data_path=data_path, batch_size=batch_size)
    net = model.model(device, beta=beta, spike_grad=spike_grad)
    loss_fn = SF.ce_rate_loss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    print('Pretraining has Begun')
    model.training(net, optimizer=optimizer, loss_fn=loss_fn, device=device,
                   train_loader=train_loader, test_loader=test_loader, num_steps=num_steps, batch_size=batch_size)
    plot_weight_heatmap(net, "Weight Heatmaps of Original Netork")
    print('Simulating Stroke')
    mask = prune_weights(net, pruning_amount=0.5)
    optimizer_masked = MaskedAdam(
        net.parameters(), mask, lr=0.001, betas=(0.9, 0.999))
    test_accuracy(net, num_steps, test_loader, device, batch_size)
    plot_weight_heatmap(net, "Weight Heatmaps of Original Netork")
    print('retraining model')
    model.training(net, optimizer=optimizer_masked, loss_fn=loss_fn, device=device,
                   train_loader=retrain_loader, test_loader=retest_loader, num_steps=num_steps, batch_size=batch_size)
    plot_weight_heatmap(net, "Weight Heatmaps of Original Netork")
