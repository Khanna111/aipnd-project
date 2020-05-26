# ------------------------------------------------------------------------------------------------------------------------
# SAMPLE COMMAND LINE ARGUMENTS
# python predict.py --checkpoint="./checkpoints/checkpoint_cl.vgg19.pth" --topk=3 'flowers/valid/16/image_06671.jpg'
# python predict.py --checkpoint="./checkpoints/checkpoint_cl.densenet121.pth" --topk=3 'flowers/valid/16/image_06671.jpg'
# ------------------------------------------------------------------------------------------------------------------------


# Imports here

import plumbing as lp
import nn_utility as lnn

import time
from collections import OrderedDict


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
    description = 'PARSER: predict.py',
    epilog="FOR EG (TO USE VGG19): python predict.py 'flowers/valid/26/image_06506.jpg' --topk=2 --checkpoint=./checkpoints/checkpoint_cl.vgg19.pth"
)

parser.add_argument('image_path',
                    action="store")
parser.add_argument('--checkpoint',
                    action="store", default="./checkpoints/checkpoint_cl.vgg19.pth")
parser.add_argument('--topk',
                    action="store", default=5, type=int)
parser.add_argument('--category_names', 
                    action="store", default='cat_to_name.json')
parser.add_argument('--gpu', 
                    action="store", default='cuda')


args                = parser.parse_args()



image_path            = args.image_path
checkpoint            = args.checkpoint
topk_arg              = args.topk
category_names        = args.category_names

print("\nDEBUG....")
for arg in vars(args):
    print (arg, getattr(args, arg))
print("---------------------------\n")



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = lp.process_image(image_path)
    image = image.numpy()
    image = torch.from_numpy(np.array([image])).float()
    
    model.eval()

    device = "cuda"

    model = model.to(device)
    image = image.to(device)

    
    with torch.no_grad():
        answer = model.forward(image)
    
    prob = torch.exp(answer)
    
    topk, indices = prob.topk(topk, dim=1 )
    print(topk)
    print(indices)
    
    return topk, indices


if __name__== "__main__":
    model_from_checkpoint, class_to_idx = lnn.load_checkpoint(checkpoint)

    topk, indices = predict(image_path, model_from_checkpoint,topk=topk_arg)
    print(topk)
    print(indices)

    cpu_device = torch.device("cpu")
    topk = topk.to(cpu_device)
    indices = indices.to(cpu_device)
    
    indices = indices.view(-1)
    print(indices.shape)
    topk = topk.view(-1)
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    print("Total keys: {}".format(len(cat_to_name)))

    idx_to_class = { label:folder for folder, label in class_to_idx.items() }

    list_cat_names_from_indices = [cat_to_name[idx_to_class[x]] for x in indices.numpy()]

    i = 0
    print (topk.numpy().size)
    while i < topk.numpy().size:
        print(f"Category Name: {list_cat_names_from_indices[i]}.... probability of {topk[i]}...")
        i += 1
  
    print("True Value from Label ")
    print("----------------------")
    folder, cat_name =  lp.image_class_and_category(image_path, cat_to_name)

        
  