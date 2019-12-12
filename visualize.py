from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
from gradcam import *
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-w","--weights",default='default',type=str,help="Pre-trained weight file (*.pth)")
parser.add_argument("-i","--input",default='default',type=str,help="Directory for an input image")
parser.add_argument("-o","--output",default='default',type=str,help="Directory for the visualization result")
args = parser.parse_args()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = list()

with open('./mosquitos_words.txt') as lines:
    for line in lines:
        line = line.strip().split(' ', 1)[1]
        line = line.split(', ', 1)[0].replace(' ', '_')
        classes.append(line)

if (os.path.isdir(args.output)):
    print('Output directory exists')
else:
    os.mkdir(args.output)

model = models.vgg16(pretrained=True)
model.classifier[6] = torch.nn.Linear(4096,6)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.weights))
model = model.module
model.eval()
model.to(device)

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

### Target layers (VGG16)
targets = [
    "features.0",
    "features.2",
    "features.5",
    "features.7",
    "features.10",
    "features.12",
    "features.14",
    "features.17",
    "features.19",
    "features.21",
    "features.24",
    "features.26",
    "features.28",
]

image_num = 0
input_image = args.input

# =========================================================================
print('='*20)
# =========================================================================

print(f"Processing image \'{input_image}\'")

raw_image = cv2.imread(input_image)[..., ::-1]
im_pil = Image.fromarray(raw_image)

image = data_transforms(im_pil).unsqueeze(0)
image_var = Variable(image)

a = transforms.ToPILImage()(image.squeeze())

tr_image = np.asarray(np.transpose(image.squeeze(0),(1,2,0)))
tr_image = cv2.cvtColor(tr_image,cv2.COLOR_BGR2RGB)
tr_image = np.uint8(normalize(tr_image)*255)
image_name = input_image.split(sep="/")[-1].split(sep=".")[-2]
save_dir = os.path.join(args.output,image_name)
if(os.path.isdir(save_dir)):
    print('Save directory exists')
else:
    os.mkdir(save_dir)
print('Visualization saved to : ',save_dir)

a.save(os.path.join(save_dir,'_input_image'+'.jpg'))
gcam = GradCAM(model=model)
probs, idx = gcam.forward(image_var.to(device))

print("-"*30)
print("|  Probability  |    Class   |")
print("-"*30)
for i in range(0, 6):
    for j in range(0,len(targets)):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer=targets[j])
        # Filename : {ClassName}_gcam_{NumLayer}
        save_gradcam(save_dir+'/{}_gcam_{}.png'.format(classes[idx[i]], targets[j]), output, tr_image)
    print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

