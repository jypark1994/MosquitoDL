
# Classification and Morphological Analysis of Vector Mosquitoes using Deep Convolutional Neural Networks  
  
## Abstract  
  Image-based automatic classification of vector mosquitoes has been investigated for decades for its practical applications such as early detection of potential mosquitoes-borne diseases. However, classification accuracy of previous approaches has never been close to human experts’ and often images of mosquitoes with certain postures and body parts, such as flatbed wings, are required to achieve good classification performance. Deep convolutional neural networks (DCNNs) are state-of-the-art approach to extracting visual features and classifying objects, and, hence, there exists great interest in applying DCNNs for the classification of vector mosquitoes from easy-to-acquire images. In this study, we investigated the capability of state-of-the-art deep learning models in classifying mosquito species having high inter-species similarity and intra-species variations. Since no off-the-shelf dataset was available capturing the variability of typical field-captured mosquitoes, we constructed a dataset with about 3,600 images of eight vector mosquito species with various postures and damage to the bodies. To further address data scarcity problems, we investigated the feasibility of transferring general features learned from generic dataset to the mosquito classification. Our result demonstrated that more than 97% classification accuracy can be achieved by fine-tuning general features if proper data augmentation techniques are applied together. Further, we analyzed how this high classification accuracy can be achieved by visualizing discriminative regions used by deep learning models. Our results showed that deep learning models exploits morphological features similar to those used by human experts.  
  
## Dataset & Weights  
  
Dataset for Train/Validation and Test are available on the link below.  
  
Images (120.8MB)  
- [Train/Val + Test](https://drive.google.com/open?id=1aIlFzGdjhu9XFQkNtdk_n8qiM88zp3XY)

>  This dataset includes about 6,000 images selected from the the raw images (approx. 3,600 images). We built Train/Val dataset using 3,000 images, and Test dataset using the rest 600 images with 5 times of data augmentation. (3,000 images). We treat the image label as the name of each separated folder. (e.g: "./TrainVal/Aedes albopictus/*.jpg")



  
Weights (720MB)  
- [Weights](https://drive.google.com/open?id=1ZzrnfPmYeaXGIbPxx2BYmLKtLxSNkqRM) (Only augmented + finetuned weights)  
  
> Weights include pytorch weight files of three architectures(ResNet50, VGG16, SqueezeNet) trained with data augmentation(aug) and finetuning(pt) method.  

|Model		 |Weight Name		|Test Accuracy(%)	|
|:------:|:--------------------:|:-----------:|
|SqueezeNet	|squeezenet_aug_pt.pth	|90.71	|
|ResNet50	|resnet50_aug_pt.pth	|96.86	|
|VGG16	 |vgg16_aug_pt.pth	|**97.74**	|

1. Download .zip files from the link above.  
  
2. Unpack the zip files on the root of the project folder.  
  
> To be like following : `./{Project_Root}/Weights, ./{Project_Root}/Datasets`

### Raw mosquito images
- [Raw Images (2.5GB)](https://drive.google.com/open?id=1XW1vrNSmNbXOqC9BoXbSIkDJlRtd-lOP)

> Including the mosquito images without data augmentation, cropping.

> Filename : {BRIGHTNESS(REL)}\_{Abb. SPECIES}\_{INDEX}.JPG (e.g: 0_Ak_1.JPG)

|Species		 |Potential Vector Disease		| Captured Location |# of images	|
|:------:|:--------------------:|:-----------:|:-----------:|
|*Ae. albopictus*	|Zika, Dengue	|Jong-no Seoul	|600| 
|*Ae. vexans*	|Zika, Westnile virus	|Tanhyeon Paju	|591|
|*Anopheles* spp.	 |Malaria	|Tanhyeon Paju	|593|
|*Cx. pipiens*	 |Westnile virus	|Jong-no Seoul	|600|
|*Cx. tritaeniorhynchus*	 |Japanese Encephalitis	|Tanhyeon Paju	|594|
|:------:|:--------------------:|:-----------:|:-----------:|
|*Ae. dorsalis*	 |-	|Tanhyeon Paju	|200|
|*Ae. koreikus*	 |-	|Tanhyeon Paju	|200|
|*Cx. inatomii*	 |-	|Tanhyeon Paju	|200|

## Dependencies  
python3>=3.7.x  
  
pytorch>=1.1.0

cv2>=4.1.2  
  
## Usage  
1. trainer.py : Train model and save weights(*.pth).  
  
   `python3 trainer.py -a [augmented] -p [pretrained] -r [learning_rate] -n [num_workers] -b [batch_size] -m [model_name] -e [num_epochs] -d [data_folder]  -t [output_folder]`  
     
   -a : Set data augmentation. (Default : 1)  
  
   > 1 (Enable) or 0 (Disable)  
  
   -p : Use pretrained weights. (Default : 1)  
  
   > 1 (Enable finetuning) or 0 (Training from the scratch)  
  
   -r : Set learning rate (Default : 5e-6)  
  
   -n : Set number of workers (Default : 2)  
  
   -b : Set batch size (Default : 32)  
  
   -m : Set the model to be trained (Default : vgg16)  
  
   > Supports vgg16, resnet50 and squeezenet.  
  
   -e : Set number of epochs (Default : 100)  
  
   -d : Set data folder for Train/Validation. (Default : "./Datasets/TrainVal")  
  
   -t : Set output folder for saving weights. (Default : "./Default")  
  
   > Training log will be saved in {OutFolder}/log/*.txt  
  
   > Trained weight will be saved in {OutFolder}/pth/*.pth  
  
  
2. tester.py : Test model, print confusion matrix, and save misclassification cases.  
  
   `python3 tester.py -q [quite_mode] -m [model_name] -w [weight_file] -t [test_folder]`  
  
   -q : Enable quite mode (Default : 1)  
  
   -m : Model to use (Default : vgg16)  
  
   > Supports vgg16, resnet50, squeezenet  
  
   -w : Set weight to use (Default : "./Weights/vgg16_aug_pt.pth")  
  
   -t : Set test dataset folder (Default : "./Datasets/Test")  
  
  
3. visualize.py : Visualize feature activation and print probability for each class.  
- Only supports vgg16 models.  (The model that we used in the visual analyzation section)
  
   `python3 visualize.py -w [weight_file] -i [input_image] -o [output_folder]`
     
   -w : Set trained weight (*.pth)  
     
   -i : Set an input image to visualize feature activation.  
     
   -o : Set the output folder for visualized feature activations.  
     
## Quick Start  
1. To train the VGG16 model with data augmentation and fine-tuning.  
- Batch size and the number of workers can be varied according to CPU/GPU preferences.  
  
   `python trainer.py -a 1 -p 1 -l 5e-6 -n 2 -b 32 -m vgg16 -e 100 -d "./Datasets/TrainVal/" -t "./Default/"`  
  
2. To evaluate the model above after training.  
- Misclassification images will be saved in "./[weight_name]_mis". (e.g: "./vgg16_aug_pt_mis")  
  
   `python tester.py -q 1 -m vgg16 -w "./Weights/vgg16_aug_pt.pth"`  
  
3. To visualize feature activation with the model above. (Supports VGG16 as the paper)  
- Extracts the highlighted feature maps of convolutional layers that contributed for the class prediction.  
  
- Visualized images will be saved in "./[output_folder]/[image_name]/..." (e.g: "./results/0_0_Aa_1/...")  
- Filename: {TargetClass}\_gcam\_{NumLayer}.jpg  
     
   `python visualize.py -i "./Datasets/Test/Aedes albopictus/0_0_Aa_1.JPG" -o "./results" -w "./Weights/vgg16_aug_pt.pth"`
