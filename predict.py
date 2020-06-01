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
import json

def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Prediction Settings")

    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str', default = "vgg16")
    
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to image file for prediction.',
                        required=True)
    # Load checkpoint
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        default="checkpoint.pth")
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.', default=5)
    
    # Import category names
    parser.add_argument('--cat_names', 
                        type=str, 
                        help='Categories to names mapping', default = 'cat_to_name.json')
    # GPU Option
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU for calculations')
    # Parse args
    args = parser.parse_args()
    
    return args
def load_checkpoint(file):
    file = args.checkpoint
    arch = args.arch
    checkpoint = torch.load(file)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['arch'])
    
    #for param in model.parameters(): param.requires_grad = False
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.epochs = checkpoint['epochs']
    model.classifier = checkpoint['hidden_layers'] 
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['model_index']
    
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = args.image
    # TODO: Process a PIL image for use in a PyTorch model
    size = 256, 256
    im = Image.open(image)
    #im = image
    
    w, h = im.size
    if w < h: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
    
    im.thumbnail(size=resize_size)
        
    l = (w - 224)/2
    u = (h - 224)/2
    r = (w + 224)/2
    b = (h + 224)/2
    
    im.crop((l,u,r,b))
    
    np_image = np.array(im)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np.transpose(np_image, (2,0,1))
    
    return np_image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose(1, 2, 0)
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
def predict(image, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    topk = args.top_k
    model.eval()
    model.to('cuda')
    
    with torch.no_grad():
        tensor = torch.from_numpy(image).type(torch.cuda.FloatTensor)
        unsqueezed = tensor.unsqueeze_(0)
        resized = unsqueezed.resize_(32,3,224,224)
        output = model.forward(resized)
        #output = output.resize_(64, 3,96,96) 
        ps = torch.exp(output)
        
    ps_top = ps.topk(topk)[0]
    idx_top = ps.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    ps_top = np.array(ps_top)[0]
    idx_top = np.array(idx_top[0])
    
    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    idx_to_class = {x: y for y, x in class_to_idx.items()}
    # Converting index list to class list
    classes_top = []
    for idx in idx_top:
        classes_top += [idx_to_class[idx]]
        
    return ps_top, classes_top
args = arg_parser()
image = process_image(args.image)
topk = args.top_k
model = load_checkpoint(args.checkpoint)
cat_names = args.cat_names
with open(cat_names, 'r') as f:
    cat_to_label = json.load(f)
    
ps_top, classes_top = predict(image, model, topk)

labels = []
for i in classes_top:
    labels.append(cat_to_label[i])

if topk == 1:
    print(f"This flower is most likely to be a: '{labels[0]}' with a probability of {round(ps_top[0]*100,4)}% ")
else:
    print(ps_top)
#print(classes_top)
    print(labels)
