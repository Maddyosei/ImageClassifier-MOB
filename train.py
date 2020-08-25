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


import argparse
import json

parser = argparse.ArgumentParser(description='flowers')

parser.add_argument('--arch', type=str, default='DenseNet', action="store", help='architecture [available: densenet, vgg]')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
parser.add_argument('--lr', type=float, default=0.003, help='Learning rate for optimizer')
parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--gpu', type=bool, default=False, help='Utilize gpu to train')

args = parser.parse_args()

if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.lr:
    learning_rate = args.lr
if args.epochs:
    epochs = args.epochs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load data 
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {'traindir' : transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])
                                                   ]),
                   'validdir' : transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])
                                                   ]),
                   'testdir' : transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])
                                                  ]),
                  }

image_datasets = {
    'traindir': datasets.ImageFolder(train_dir, transform=data_transforms['traindir']),
    'validdir': datasets.ImageFolder(valid_dir, transform=data_transforms['validdir']),
    'testdir': datasets.ImageFolder(test_dir, transform=data_transforms['testdir']), 
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'traindir': torch.utils.data.DataLoader(image_datasets['traindir'], batch_size=64, shuffle=True),
    'validdir': torch.utils.data.DataLoader(image_datasets['validdir'], batch_size=64),
    'testdir': torch.utils.data.DataLoader(image_datasets['testdir'], batch_size=64)
}

datasets_sizes = {
    'traindir': len(image_datasets['traindir']),
    'validdir': len(image_datasets['validdir']),
    'testdir': len(image_datasets['testdir'])
}

class_names = image_datasets['traindir'].classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def create_model():   
    model = models.densenet121(pretrained=True)
    
    print("Our model: \n\n", model)
    print("The state dict keys: \n\n", model.state_dict().keys())
    
    #Freeze parameters and save the checkpoint
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, args.hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(args.hidden_units, 200)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(200, 102)),
        ('output', nn.LogSoftmax(dim=1))
                                            ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.lr)
    
    return model, criterion, optimizer
    
model, criterion, optimizer = create_model()

def validation(model, dataloaders, criterion):

    accuracy = 0
    test_loss = 0
    
    for images, labels in dataloaders['validdir']:
        images = Variable(images)
        labels = Variable(labels)        
        images, labels = images.to(device), labels.to(device)
        
        logps = model.forward(images)
        test_loss += criterion(logps, labels).item()
        
        # Calculate accuracy
        ps = torch.exp(logps)
        equals = (labels.data == ps.max(1)[1])
        accuracy += equals.type_as(torch.FloatTensor()).mean()
        
    return test_loss, accuracy    

def train(model, dataloaders, epochs, print_every, criterion, optimizer, device):
       
    steps = 0
    running_loss = 0
    for e in range(epochs):
        model.train()
        model.to(device)
        
        for images, labels in dataloaders['traindir']:
            images, labels = Variable(images), Variable(labels)
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                model.to(device)
                
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloaders, criterion)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['validdir'])),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['validdir'])))
                
                running_loss = 0
                model.train()
    return model
    
model_trained = train(model, dataloaders, args.epochs, 40, criterion, optimizer, device)

def saved_model(model_trained):
    model.class_to_idx = image_datasets['traindir'].class_to_idx
    idx_to_class = {v : k for k, v in model.class_to_idx.items()}
    model.name = arch
    
    checkpoint = {'input_size': args.hidden_units,
                  'output_size': 102,
                  'batch_size': 64,
                  'data_transforms': data_transforms,
                  'arch': model.name,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': image_datasets['traindir'].class_to_idx}
    
    torch.save(checkpoint, args.save_dir)    

saved_model(model_trained)
