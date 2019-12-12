import os
import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time
import argparse
from PIL import Image
from torch import nn
from torchvision import models, transforms, utils

parser = argparse.ArgumentParser()
parser.add_argument("-q","--quiet",default=1,type=int,help="If enabled(1), the tester only shows the matrix and accuracy.")
parser.add_argument("-m","--model",default="vgg16",type=str,help="Model name")
parser.add_argument("-w","--weight",default="./Weights/vgg16_aug_pt.pth")
parser.add_argument("-t","--testset",default="./Datasets/Test/",type=str,help="Location of testing dataset")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.model == "vgg16":
    model = models.vgg16(pretrained=False)
    model.eval()
    model = torch.nn.DataParallel(model)

    num_ftrs = model.module.classifier[6].in_features
    model.module.classifier[6] = torch.nn.Linear(num_ftrs, 6)
elif args.model == "resnet50":
    model = models.resnet50(pretrained=False)
    model.eval()
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    num_ftrs = model.module.fc.in_features
    model.module.fc = nn.Linear(num_ftrs, 6)
elif args.model == "squeezenet":
    model = models.squeezenet1_0(pretrained=False)
    model.eval()
    model = torch.nn.DataParallel(model)
    
    model.module.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512,6,kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(13,stride=1)
    )
    model.module.num_classes = 6
else:
    print("Model ", args.model, " not supported.")
    exit()

target_model = args.weight

model.load_state_dict(torch.load(target_model))
model.eval()
model.to(device)
print("="*50)
print("Testing the model : {}".format(target_model))
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

classes = {}

with open('./mosquitos_words.txt') as lines:
    for idx, line in enumerate(lines):
        line = line.strip().split(' ', 1)[1]
        line = line.split(', ', 1)[0]
        classes[idx] = line

im_dir = args.testset
scores = np.zeros((6,6))
partial_img_cnt = 0
total_img_cnt = 0

start = time.time()
correct_count = 0
input_name = args.weight.split(sep='/')[-1].split(sep='.')[-2]
mis_dir = os.path.join("./",input_name+"_mis")

if(os.path.isdir(mis_dir) == False):
    os.mkdir(mis_dir)
for key, value in classes.items():
    class_dir = im_dir + value
    im_list = os.listdir(class_dir)
    if(args.quiet == 0):
        print("Processing ... ", value)
    class_count = np.zeros(6)
    for idx, img in enumerate(im_list):
        if img.split('.')[1] == 'txt':
            continue
        image_name = class_dir+"/"+img
        raw_image = cv2.imread(image_name)[..., ::-1]
        im_pil = Image.fromarray(raw_image)
        image = data_transforms(im_pil).unsqueeze(0)

        pred_list = model(image)
        pred_list = pred_list.cpu().detach().numpy()[0]
        pred = np.argmax(pred_list)
        total_img_cnt = total_img_cnt + 1
        if(pred != key): # Misclassfication cases
            pilImage = transforms.ToPILImage()(image.squeeze(0))
            mis_img_dir = os.path.join(mis_dir, classes[key])
            if(os.path.isdir(mis_img_dir) == False):
                os.mkdir(mis_img_dir)
            pilImage.save(os.path.join(mis_img_dir,
                          str(partial_img_cnt) + "_" + value + "_to_" + classes[pred] +'.png'))
        else:
            correct_count = correct_count+1
        class_count[pred] = class_count[pred]+1
        partial_img_cnt = partial_img_cnt + 1
    class_scores = (class_count/partial_img_cnt)*100
    scores[key] = class_scores
    partial_img_cnt=0
test_accuracy = correct_count/total_img_cnt
scores = np.around(scores,decimals=2)
print("----- Confusion Matrx (%) -----")
print(scores)
print("-------------------------------")
print("Correct {} of {}.".format(correct_count,total_img_cnt))
print("Test Accuracy : {:.2f}%".format(test_accuracy*100))
end = time.time()
    
print("Elapsed time : {:.2f} [sec]".format(end-start))

print("="*50)
