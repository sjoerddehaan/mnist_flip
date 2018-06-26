## Code for notebook flipper.py
## In the notebook, run %run flipper.py

import sys
sys.path.append("../")
from flipper.transform import PinballFlip, duplicate_channels, plot_random_flips
from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import os
import copy
import torchvision
from torchvision import datasets, transforms
PATH = '~/data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

flipper = PinballFlip(p=0.5)
image_transforms = transforms.Compose(
        [transforms.Resize(size=224),
         flipper.image_transform,
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         duplicate_channels
         ])


target_transforms = flipper.target_transform
trainset = torchvision.datasets.MNIST(root=PATH, train=True,
                                            download=True, transform=image_transforms,
                                            target_transform=target_transforms)
valset = torchvision.datasets.MNIST(root=PATH, train=False,
                                            download=True, transform=image_transforms,
                                            target_transform=target_transforms)
image_datasets = {'train': trainset,
                  'val': valset}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = ('Flipped', 'Original')


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """ Train PyTorch model
        Source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model