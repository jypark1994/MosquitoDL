import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy
import datetime
import argparse
from operator import eq
from torch.utils.data import SubsetRandomSampler

parser = argparse.ArgumentParser()
parser.add_argument("-a","--augmentation",default=0,type=int,help="Augmentation for dataset")
parser.add_argument("-p","--pretrained",default=0,type=int,help="Enable pretrained weights from imagenet") 
parser.add_argument("-r","--learningrate",default=0.00001,type=float,help="Learning rate for optimizer")
parser.add_argument("-n","--numworkers",default=4,type=int,help="Num. of workers for the dataloader")
parser.add_argument("-b","--batchsize",default=16,type=int,help="Batch size for training")
parser.add_argument("-m","--model",default="vgg16",type=str,help="Choose vgg16, resnet152 or squeezenet")
parser.add_argument("-e","--epochs",default=100,type=int,help="Num. of epochs for training")
parser.add_argument("-d","--datafolder",default="./Datasets/TrainVal",type=str,help="Specify the training data folder")
parser.add_argument("-t","--rootdir",default="./default/",type=str,help="Specify root directory for the results")
args = parser.parse_args()

print(args.rootdir)
if(not os.path.isdir(args.rootdir)):
    os.mkdir(args.rootdir)
else:
    pass

log_dir = os.path.join(args.rootdir,"log")
pth_dir = os.path.join(args.rootdir,"pth")

if(not os.path.isdir(log_dir)):
    os.mkdir(log_dir)

if(not os.path.isdir(pth_dir)):
    os.mkdir(pth_dir)

if args.pretrained==1:
    pt_str = "pt" 
    pt = True
else:
    pt_str = "npt"
    pt = False

if args.augmentation==1:
    aug_str = "aug"
else:
    aug_str = "org"

log_file = log_dir + "/" + args.model + "_" + aug_str + "_" + pt_str +  ".txt"
pth_file = pth_dir + "/" + args.model + "_" + aug_str + "_" + pt_str +  ".pth"

log = open(log_file, "a+")

init_scale = 1.15

# Data augmentation for training
if args.augmentation == 1:
    data_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
            transforms.RandomAffine(360,scale=[init_scale-0.15,init_scale+0.15]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
    ])
else:
    data_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
    ])

data_dir = args.datafolder
image_datasets = datasets.ImageFolder(os.path.join(data_dir),transform=data_transforms)

class_names = image_datasets.classes
dataset_sizes = [596,596,596,596,596]
training_set = torch.utils.data.random_split(image_datasets,[596,596,596,596,596])

dataloaders = {x: torch.utils.data.DataLoader(training_set[x], batch_size=args.batchsize,
                                             shuffle=True, num_workers=args.numworkers)
                for x in range(5)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_, classes = next(iter(dataloaders[0]))

# We use 100 of 500 images in each folds for validation set. Rest 400 images are training set
tr_sequence = [[1,2,3,4,0],[0,2,3,4,1],[0,1,3,4,2],[0,1,2,4,3],[0,1,2,3,4]]
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    # 1. Average validation accuracy for each epochs.
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_since = time.time()
        if(epoch%10 == 0):
            print("Checkpoint saved ... Epoch : ", epoch)
            torch.save(model_ft.state_dict(), pth_file.replace(".pth","_"+str(epoch)+".pth"))
        scheduler.step() # A step : An epoch
        print("----- Epoch : {} -----".format(epoch))
        for param_group in optimizer.param_groups:
            # Current learning rate ...
            print("Learning rate : ",param_group['lr'])

        epoch_val_sum = 0

        for i in range(5): # 5 folds...
            print("---> Fold : {}".format(i))
            for f in tr_sequence[i]: # Iterate for folds...
                if f is not i: # Training Set
                    model.train()  # Set model to training mode
                else: # Validation Set
                    model.eval()   # Set model to evaluation mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data for current set of the fold.
                for inputs, labels in dataloaders[f]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(f is not i):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if f is not i:
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[f]
                epoch_acc = running_corrects.double() / dataset_sizes[f]
                
                if(f == i):
                    print('Validation({}) - Loss: {:.4f}, Acc: {:.4f}'.format(
                        f, epoch_loss, epoch_acc))
                    epoch_val_sum = epoch_val_sum + epoch_acc
                else:
                    print('Training({}) - Loss: {:.4f}, Acc: {:.4f}'.format(
                        f, epoch_loss, epoch_acc))
                # deep copy the model

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_since
        print("-"*20)
        print('Epoch time : {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))

        estimated_time = (num_epochs-epoch)*epoch_time
        print('Estimated time ({:.0f}/{:.0f}) : {:.0f}m {:.0f}s'.format(
            epoch, num_epochs, estimated_time // 60, estimated_time % 60))
        
        epoch_val = epoch_val_sum/5
        if epoch_val > best_acc:
            best_acc = epoch_val
            best_model_wts = copy.deepcopy(model.state_dict())
        print("Epoch {} accuracy : {}".format(epoch,epoch_val))
        log.write("{}\n".format(epoch_val))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

if args.pretrained==1:
    pt = True
else:
    pt = False

print("Using ",torch.cuda.device_count(),"GPU.")
if args.model == "vgg16":
    model_ft = models.vgg16(pretrained=pt)

    model_ft = torch.nn.DataParallel(model_ft)
    model_ft.to(device)

    num_ftrs = model_ft.module.classifier[6].in_features
    model_ft.module.classifier[6] = torch.nn.Linear(num_ftrs, 6)
elif args.model == "resnet50":
    model_ft = models.resnet50(pretrained=pt)

    model_ft = torch.nn.DataParallel(model_ft)
    model_ft.to(device)
    
    num_ftrs = model_ft.module.fc.in_features
    model_ft.module.fc = nn.Linear(num_ftrs, 6)
elif args.model == "squeezenet":
    model_ft = models.squeezenet1_0(pretrained=pt)

    model_ft = torch.nn.DataParallel(model_ft)
    model_ft.to(device)
    
    model_ft.module.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512,6,kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(13,stride=1)
    )
    model_ft.module.num_classes = 6
else:
    print("Model ", args.model, " not supported.")
    exit()

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.learningrate)
# Decay LR by a factor of 0.75 every 15 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.75)

print("-"*20)

if pt == True:
    print("Finetuning all layers")
else:
    print("Training from scratch") 

print("Filename : {}".format(pth_file))
print("Model : ", args.model)
print("-"*20)

log.write("*"*20+"\n")

model_ft = model_ft.cuda()
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    num_epochs=args.epochs)

torch.save(model_ft.state_dict(), pth_file)
print("Model state saved to : " + pth_file)
print("-"*20)
