# Import modules
import argparse
import json
import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
import numpy as np
import pandas as pd
from PIL import Image

# Command line argument fetching
parser = argparse.ArgumentParser('This script can be used to predict the class of a flower using trained model.')
parser.add_argument('input_img', metavar = 'input_img', type = str, help = 'Path of Testing image.')
parser.add_argument('checkpoint', metavar = 'checkpoint', type = str, help = 'Path of Checkpoint.')
parser.add_argument('--topk', type = int, default = 1, help = 'Top k most likely classes')
parser.add_argument('--category_names', type = str, default = '/home/workspace/ImageClassifier/cat_to_name.json')
parser.add_argument('--gpu', type = int, default = 1, help = 'GPU enabled (integer): 1:True, 0:False')
args = parser.parse_args()

# Command line arguments stored below
img_path = args.input_img
checkpoint_path = args.checkpoint
topk = args.topk
category_names = args.category_names
gpu_enabled = True if args.gpu == 1 else False

# Enabling gpu if available
device = torch.device("cuda" if gpu_enabled else "cpu")

# Category to Name mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# Loading checkpoint
def load_checkpoint(path):
	checkpoint = torch.load(path)
	
	arch = checkpoint['model_arch']
	
	if arch==2:
		model = models.vgg19(pretrained=True)
	else:
		model = models.resnet101(pretrained=True)

	for param in model.parameters():
		param.requires_grad = False
	
	if arch==2:
		model.classifier = checkpoint['classifier']
	else:
		model.fc = checkpoint['classifier']
	
	model.class_to_idx = checkpoint['class_to_idx']
	model.load_state_dict(checkpoint['model_state_dict'])
    
	return model
	
model = load_checkpoint(checkpoint_path)

# Function to preprocess image before prediction
def process_image(image):
    
    pil_img = Image.open(image)
    wid, hgt = pil_img.size
    size = 256, 256
    crop_size = 224
    
    if wid > hgt:
        pil_img.thumbnail((20000, size[1]))
    else:
        pil_img.thumbnail((size[0], 20000))
        
    left = (size[0] - crop_size)/2
    top = (size[1] - crop_size)/2
    right = (left + crop_size)
    bottom = (top + crop_size)
    pil_img = pil_img.crop((left, top, right, bottom))
    
    np_image = np.array(pil_img)/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean)/std
    
    final_img = np_image.transpose(2,0,1)
    return final_img

# Prediction function for the top - k classes along with probability for flower in test image using trained model
def predict(image_path, model, topk=topk):
    
	model.to(device)
	model.eval()
    
	pytorch_img = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to(device)
	logps = model.forward(pytorch_img)
	ps = torch.exp(logps)
	top_p, top_class = ps.topk(topk, dim=1)
	
	top_p = top_p.cpu().detach().numpy().tolist()[0]
	top_class = top_class.cpu().tolist()[0]
	#print(top_p)
	#print(top_class)
	labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(cat_to_name)})
	labels = labels.set_index('class')
    
	labels = labels.iloc[top_class]
	labels['predictions'] = top_p
    
	return labels

# Predict the flower with class probability
print('Predicting top - {} class(es) and class probabilities. GPU enabled? - {}'.format(topk,gpu_enabled))
labels = predict(img_path, model) 

print("\nPredicted flower along with Probability - \n", labels)

print("\npredict.py has finished execution")
