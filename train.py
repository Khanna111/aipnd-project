# ------------------------------------------------------------------------------------------------------------------------
# SAMPLE COMMAND LINE ARGUMENTS
# python train.py  --arch="vgg19" --epoch=3 --save_dir="./checkpoints"  './flowers'
# python train.py  --arch="densenet121" --epoch=3 --save_dir="./checkpoints"  './flowers'
# ------------------------------------------------------------------------------------------------------------------------



# Imports here

import plumbing as lp
import nn_utility as lnn

import os

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
import argparse

arch = {"vgg19":25088,
        "densenet121":1024}

parser = argparse.ArgumentParser(
    description = 'PARSER: train.py',
    epilog =  "FOR EG (TO USE VGG19): python train.py  --arch='vgg19' --epoch=3 --save_dir='./checkpoints'  './flowers'" 
)

# data_dir="./flowers"

parser.add_argument('data_dir',
                    action="store", default="./flowers")
parser.add_argument('--save_dir',
                    action="store", default="./checkpoints")
parser.add_argument('--arch',
                    action="store", default="vgg19", choices=["vgg19", "densenet121"])
parser.add_argument('--hidden_units',
                    action="store", dest="hidden_units", type=int)
parser.add_argument('--learning_rate',
                    action="store", type=float,default=0.001)
parser.add_argument('--epochs',
                    action="store", default=2, type=int)
parser.add_argument('--dropout',
                    action="store", type=float, default=0.2)
parser.add_argument('--gpu',
                    action="store_true", default="gpu")

args                = parser.parse_args()



data_dir            = args.data_dir
checkpoint_path     = args.save_dir
arch                = args.arch
hidden_units        = args.hidden_units
learning_rate       = args.learning_rate
epochs              = args.epochs
dropout             = args.dropout
gpu                 = args.gpu


if gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

if arch=="vgg19":
    input= 25088
    output=102
    if hidden_units == None:
        hidden_units = 4096
elif arch=="densenet121":
    input=1024
    output=102
    if hidden_units == None:
        hidden_units = 500



        
print("\nDEBUG....")
for arg in vars(args):
    print (arg, getattr(args, arg))
checkpoint_file = checkpoint_path+"/checkpoint_cl."+arch+".pth"
print("checkpoint: ", checkpoint_file)
print("Hiden Units: ", hidden_units)
print("---------------------------\n")

print(f"Creating SAVE_DIR: '{checkpoint_path}' if does not exist")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)






def train_model(model, criterion, optimizer, num_epochs=5, print_every_step=5, device_name='cuda'):
    epochs = num_epochs
    print_every = print_every_step
    device = device_name

    trainloader, c_i = lp.get_loader('train', data_dir)
    testloader, c_i = lp.get_loader('test', data_dir)

    s_time = time.time()
    e_time = s_time + 120

    steps = 0
    running_loss = 0

    model.train()

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
#                 break
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()
    e_time = time.time() - s_time

    print('Finished in {:.0f}min, {:.0f}secs'.format(e_time // 60, e_time % 60))

    return model
    
    
if __name__== "__main__":

    model, criterion, optimizer = lnn.get_model(arch,"cuda",input,hidden_units,output,dropout,learning_rate)
    
    from workspace_utils import active_session

    with active_session():
        model_trained = train_model(model,criterion,optimizer,epochs,5,device )

    print(f"Epochs: {epochs}.. arch: {arch}.. ")

    tloader, class_to_idx = lp.get_loader("train", data_dir)

    # Create checkpoint save_dir if not exists:
    
   
        
    # TODO: Save the checkpoint
    checkpoint = {
        'arch': arch,
        'num_epochs': epochs,
        'input_size': input,
        'output_size': output,
        'hidden_layers': [hidden_units],
        'dropout': dropout,
        'class_to_idx': class_to_idx,
        'optim_state_dict': optimizer.state_dict(),
        'state_dict': model_trained.state_dict()}

    torch.save(checkpoint,checkpoint_file)
    
    print(f"Checkpoint Saved: {checkpoint_file}")
