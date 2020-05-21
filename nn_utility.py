import time
from collections import OrderedDict

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import json
from PIL import Image


def get_model(name="vgg19", device_name="cuda", input=25088, hidden=4096, output=102, dropout=0.2, learning_rate=0.001):

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif name == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)


    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input, hidden)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),

        #                           ('fc2', nn.Linear(1568, 1568)),
        #                           ('relu2', nn.ReLU()),
        #                           ('dropout2', nn.Dropout(0.2)),

        #   ('fc2', nn.Linear(3136, 1568)),
        #   ('relu2', nn.ReLU()),
        #   ('dropout2', nn.Dropout(0.2)),

        #                           ('fc3', nn.Linear(4096, 784)),
        #                           ('relu3', nn.ReLU()),
        #                           ('dropout3', nn.Dropout(0.2)),

        ('fc4', nn.Linear(hidden, output)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    return model, criterion, optimizer


