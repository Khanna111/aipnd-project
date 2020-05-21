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

def get_loader(name='train', data_dir='flowers'):


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)

    # This tells us what label (index) is mapped to what folder name (class)
    # The keys are the classses (folder names)
    # The label is the index which is used in the model and everywhere
    class_to_idx = train_data.class_to_idx

    number_of_train_images = len(train_data)
    class_names = train_data.classes
    number_of_classes = len(class_names)
    class_to_idx = train_data.class_to_idx
    print("class_names = ", class_names)
    print('---------------------------------------')
    print("number_of_classes", number_of_classes)
    print('---------------------------------------')
    print("number_of_train_images = ", number_of_train_images)
    print('---------------------------------------')
    print("class_to_idx = ", class_to_idx)
    print('---------------------------------------')
    print("class_to_idx is a : ", type(class_to_idx))

    if name=='train':
        return trainloader, class_to_idx
    elif name=='test':
        return testloader, class_to_idx
    else:
        return validationloader, class_to_idx