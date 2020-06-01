# Imports here
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import random, os

def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Training Settings")
    
    # model architecture
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str', default = "vgg16")
    
    # checkpoint directory
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Directory to save checkpoints as str', default='checkpoint.pth')
    
    # hyperparameters
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Learning rate as float', default = .0005)
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for classifier as int', default = 1000)
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int', default = 5)
    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU for calculations', default = "gpu")
    
    # Parse args
    args = parser.parse_args()
    return args
args = arg_parser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, .0225])])
train_transforms = transforms.Compose([transforms.RandomRotation(35),
                                       transforms.RandomResizedCrop(224),
                                       #transforms.RandomVerticalFlip(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, .0225])])
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
arch = args.arch
exec("model = models.{}(pretrained=True)".format(arch))
# TODO: Build and train your network
def set_up_model(arch):
    
    model.name = arch
    from collections import OrderedDict
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_feat = model.classifier[0].in_features
    hidden_units = args.hidden_units
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_feat,hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.25)),
                          ('fc2',nn.Linear(1000,102)),
                          #('fc3',nn.Linear(512,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    
    return model
model = set_up_model(arch)
lr = args.learning_rate
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
def validate_model(model, validloader, criterion):
    model.to('cuda')
    valid_accuracy = 0
    valid_loss = 0
    for ii, (inputs,labels) in enumerate(validloader):
        optimizer.zero_grad()
        
        inputs, labels = inputs.to('cuda') , labels.to('cuda')
        
        with torch.no_grad():    
            outputs = model.forward(inputs)
            valid_loss += criterion(outputs, labels)
            ps = torch.exp(outputs)
            valid_correct = (labels.data == ps.max(dim=1)[1])
            valid_accuracy += valid_correct.type(torch.FloatTensor).mean()
                   
    return valid_loss, valid_accuracy
def train_model(model, trainloader, epochs, print_every, criterion, optimizer, device='cuda'):
    epochs = epochs
    print_every = print_every
    steps = 0
    
    model.to('cuda')
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            
            outputs = model.forward(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            
            running_loss += train_loss.item()
            
            if steps % print_every == 0:
               
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_accuracy = validate_model(model, validloader, criterion)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                "Training Loss: {:.4f}".format(running_loss/print_every),
                "Valid Loss: {:.4f}".format(valid_loss/len(validloader)),
                "Valid Accuracy: {:.4f}".format(valid_accuracy/len(validloader)))
                running_loss = 0
                # Turning training back on
                model.train()
    return model
def check_accuracy_test(testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on 10000 test images: %d %%' % (100 * correct / total))
    
epochs = args.epochs
train_model(model, trainloader, epochs, 40, criterion, optimizer, 'cuda')
check_accuracy_test(testloader)
# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 4608,
              'output_size': 102,
              'epochs': epochs,
              'arch': arch,
              'hidden_layers': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'model_index' : model.class_to_idx,
             }
path = args.checkpoint
torch.save(checkpoint, path)
