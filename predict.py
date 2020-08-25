#Import statements
import numpy as np
import torch

import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from torch import tensor

import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from torch.autograd import Variable

import PIL
from PIL import Image

import time
import copy
import os

os.environ['QT_QPA_PLATFORM']='offscreen'

import argparse
import json

arch=''
image_path = 'flowers/test/1/image_06752.jpg'
save_dir = 'checkpoint.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='path of image')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
parser.add_argument('--json', type=str, default='cat_to_name.json', help='Path to classes map file')
parser.add_argument('--topk', type=int, default=5, help='Number of highest probabilities')
parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden units', type=int, default=500, help='hidden units for fc layer') 

args = parser.parse_args()

if args.save_dir:
    checkpoint = args.save_dir
if args.image_path:
    image_path = args.image_path
if args.topk:
    top_k = args.topk
if args.json:
    file_name = args.json
if args.gpu:
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(file_name, 'r') as f:
    cat_to_name = json.load(f)    

def load_checkpoint(save_dir):
    if torch.cuda.is_available():
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location='cpu')
        
    model = models.densenet121(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

model = load_checkpoint(args.save_dir)

def process_image(image_path):
   
    image = Image.open(image_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(image)
    
    return img_tensor.numpy()

processed_image = process_image(image_path)

model.class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v : k for k, v in model.class_to_idx.items()}

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # turn off drop-out
    model.eval()
    # change to cuda
    model.to('cuda')
    img_torch = torch.from_numpy(process_image(image_path))
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    print("Predicting the top {} classes with {} pre-trained model.".format(topk, model.__class__.__name__))
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
    
    # Calculate the class probabilities (softmax) for image
    probability = F.softmax(output.data, dim=1)
    
    top = torch.topk(probability, topk)
    
    probs = top[0][0].cpu().numpy()
    classes = [idx_to_class[i] for i in top[1][0].cpu().numpy()]
    
    return probs, classes

probs, classes = predict(image_path, model)
