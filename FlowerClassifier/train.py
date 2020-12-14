# Import modules
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict

# Command line argument fetching
parser = argparse.ArgumentParser('This script can be used to train a network to classify flower images.')
parser.add_argument('dataDir', metavar = 'dataDir', type = str, default = '/home/workspace/ImageClassifier/flowers', help = 'Image data root directory')
parser.add_argument('--checkpointDir', type = str, default = '/home/workspace/ImageClassifier/checkpoints', help = 'Checkpoints directory')
parser.add_argument('--arch', type = int, default = 1, help = 'Select model architecture(integer): 1:resnet101, 2:vgg19')
parser.add_argument('--lr', type = float, default = 0.003, help = 'Learning Rate')
parser.add_argument('--epochs', type = int, default = 5, help = 'Training epochs')
parser.add_argument('--hidden_units', nargs = '+', type = int, default = [1024, 512], help = 'Number of hidden units in Hidden Layer 1 and Hidden Layer 2 separated by a space.')
parser.add_argument('--gpu', type = int, default = 1, help = 'GPU enabled (integer): 1:True, 0:False')
args = parser.parse_args()

# Command line arguments stored below
dataDirectory = args.dataDir
checkpoint_dir = args.checkpointDir
arch = args.arch
learnrate = args.lr
epochs = args.epochs
hidden_units = args.hidden_units[:2]
gpu_enabled = True if args.gpu == 1 else False

# Selecting input data directory for train and validation data
data_dir = dataDirectory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Data transformations
data_transforms = [transforms.Compose([transforms.RandomRotation(20),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                   transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])]


# Training and Validation datasets initialization
image_datasets = [datasets.ImageFolder(train_dir, transform = data_transforms[0]),
                  datasets.ImageFolder(valid_dir, transform = data_transforms[1])]

# Initializing Dataloaders
dataloaders = {'trainloader' : DataLoader(image_datasets[0], batch_size = 64, shuffle = True),
              'validloader' : DataLoader(image_datasets[1], batch_size = 64)}

# Define the model
device = torch.device("cuda" if gpu_enabled else "cpu")

if arch==2:
	model = models.vgg19(pretrained=True)
else:
	model = models.resnet101(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

if arch==2:
	classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088,hidden_units[0])),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units[0],hidden_units[1])),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(0.2)),
                          ('fc3',nn.Linear(hidden_units[1],102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
	model.classifier = classifier
else:
	classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048,hidden_units[0])),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units[0],hidden_units[1])),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(0.2)),
                          ('fc3',nn.Linear(hidden_units[1],102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

	model.fc = classifier

#Classifier name is different for different architectures
def classifier_info(architecture):
	if architecture == 2:
		return model.classifier
	else:
		return model.fc

# Train the model
print('Training model - ','vgg19' if arch==2 else 'resnet101')
print('Hidden layer sizes - {}, Learning Rate - {}, Epochs - {}, GPU enabled? - {}'.format(hidden_units,learnrate,epochs,gpu_enabled))

criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier_info(arch).parameters(), lr=learnrate)
model.to(device);

epochs = epochs
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
	for images, labels in dataloaders['trainloader']:
		steps += 1
		
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()

		logps = model.forward(images)
		loss = criterion(logps, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		if steps % print_every == 0:
			val_loss = 0
			accuracy = 0
			model.eval()
			
			with torch.no_grad():
				for images, labels in dataloaders['validloader']:
					images, labels = images.to(device), labels.to(device)
					logps = model.forward(images)
					batch_loss = criterion(logps, labels)
					val_loss += batch_loss.item()

					ps = torch.exp(logps)
					top_p, top_class = ps.topk(1, dim=1)
					equals = top_class == labels.view(*top_class.shape)
					accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
				
			print("Epoch : ", epoch+1)
			print("Training loss : ", running_loss/len(dataloaders['trainloader']))
			print("Validation loss : ", val_loss/len(dataloaders['validloader']))
			print("Validation Accuracy :", 100 * accuracy/len(dataloaders['validloader']), end = '\n')
			running_loss = 0
			model.train()
                
# Save checkpoint
model.class_to_idx = image_datasets[0].class_to_idx

checkpoint = {'classifier': classifier_info(arch),
              'class_to_idx': model.class_to_idx,
              'epochs': epochs, 
              'model_state_dict' : model.state_dict(),
              'optim_state_dict': optimizer.state_dict(),
			  'model_arch': arch}

torch.save(checkpoint, checkpoint_dir + '/checkpoint2.pth')

print("train.py has finished execution")