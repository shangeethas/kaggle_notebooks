#!/usr/bin/env python
# coding: utf-8

# # About this Competition 
# 
# After Melanoma , once again this year we have been treated with a classic computer vision classification problem . Its a great oppurtunity for anyone who has just started with CV to try their hands on this Live Competition and make it their first . On top of everything the metric for this competition is classification accuracy , how often does that happen . 
# 
# In normal terms what is actually required of you is to become a Leaf doctor and help the farmers identify the infectious leaves and cure them at an affordable rate 😛
#  
# # About this Notebook
# 
# * As always this is a beginner Friendly Notebook in which I tell you , how you can efficiently become a leaf doctor with specialization in Cassava Leaf Diseases 😛 and with primary methodology being deep learning
# 
# * I will cover everything you need to know , from the specialization knowledge to methodologies with baseline examples of different ideas that I suggest for solving the problem
# 
# * Without much confusion , you can follow this notebook and make this your first Live CV competition
# 
# * If you are completely new to machine learning and kaggle have a look at this [guide](https://www.kaggle.com/tanulsingh077/tackling-any-kaggle-competition-the-noob-s-way) I have written
# 
# # Step 1 : Analyzing the Patient
# 
# * What would be a first step that a Leaf doctor should do before anything considering the fact that his client can't speak ?
# The answer is simple right , analyze what's wrong by looking at the patient
# 
# * But how does a doctor understands if something is wrong by just looking at it? 
# For this as a doctor , he should know what a normal Patient/Leaf looks like and observe deviations (in pattern ,color, texture,etc) from the normal behavior to separate the healthy patients from infected ones . Now to further classify the infected ones into specific class of diseases doctor should also know how the patient/leaf condition looks like in different diseases
# 
# With these pointers in mind let's start with basic familarity

# In[1]:


import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')


# In[2]:


# Preliminaries
import os
from pathlib import Path
import glob
from tqdm import tqdm
tqdm.pandas()
import json
import pandas as pd
import numpy as np

## Image hash
import imagehash

# Visuals and CV2
import seaborn as sn
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Keras and TensorFlow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array 
from keras.applications.resnet50 import preprocess_input 

# models 
from keras.applications.resnet50 import ResNet50
from keras.models import Model

#torch
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader


# # Utils
# 
# Section for Utility Functions

# In[3]:


def plot_images(class_id, label, images_number,verbose=0):
    '''
    Courtesy of https://www.kaggle.com/isaienkov/cassava-leaf-disease-classification-data-analysis
    '''
    plot_list = train[train["label"] == class_id].sample(images_number)['image_id'].tolist()
    
    # Printing list of images
    if verbose:
        print(plot_list)
        
    labels = [label for i in range(len(plot_list))]
    size = np.sqrt(images_number)
    if int(size)*int(size) < images_number:
        size = int(size) + 1
        
    plt.figure(figsize=(20, 20))
    
    for ind, (image_id, label) in enumerate(zip(plot_list, labels)):
        plt.subplot(size, size, ind + 1)
        image = cv2.imread(str(BASE_DIR/'train_images'/image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(label, fontsize=12)
        plt.axis("off")
    
    plt.show()


# In[4]:


BASE_DIR = Path('../input/cassava-leaf-disease-classification')

## Reading DataFrame having Labels
train = pd.read_csv(BASE_DIR/'train.csv')

## Label Mappings
with open(BASE_DIR/'label_num_to_disease_map.json') as f:
    mapping = json.loads(f.read())
    mapping = {int(k): v for k,v in mapping.items()}

print(mapping)


# <b>As we can see we have 4 diseases about which we will have to learn in this doctoe's course , we will dive deeper into each of them one by one framing our understanding about the characteristics and other things but before that lets map these disease names to labels in our dataset </b>

# In[5]:


train['label_names'] = train['label'].map(mapping)
train.head()


# ## Step 1.1 Learning about the Healthy ones
# 
# Now we have everything in one place we can start looking at the healthy images and form our understanding of characteristic of healthy cassava leaves . `Below is the image of a healthy cassava leaf from Google` 
# 
# ![](https://cdn.shortpixel.ai/client/to_avif,q_lossless,ret_img,w_795,h_532/https://organic.ng/wp-content/uploads/2017/02/CASSAVA-LEAF.jpg)
# 
# * From the above image we can say that one of the characteristic of a Healthy Cassava Leaf is that it should be fairly green and upright without much cuts,texture change , yellowish gradient ,etc
# 
# Let's now look at the healthy ones in the dataset and see if they are in close mix with the above image

# In[6]:


train[train['label_names']=='Healthy']['image_id'].count()


# * Out of 21k images only 2577 are the healthy ones , the imbalance in labels is clearly visible

# In[7]:


plot_images(class_id=4, 
    label='Healthy',
    images_number=6,verbose=1)


# Now if you run the above function three-four times and carefully observe the different images you see new images everytime , you will realize the following :
# * Not all the images have leaves close-up , some images might have the whole tree with leaves barely visible to human eye , some show more stem than leaves i.e to say the image set is fairly noisy
# * What's more surprising though is some of the images of healthy leaves look like they are infected and show a yellow or yellowish gradient type of color , which should be highly unlikely (we will have to investigate that)
# 
# ### Investigating Outliers :  
# To investigate on point 2 , I have the following Idea :
# 
# * The Idea here is to cluster the healthy Images and have a look at respective clusters to see if we can find the outlier cluster and damaged cluster.
# * We will use Resnet18 to generate features for clustering 

# In[8]:


def extract_features(image_id, model):
    file = BASE_DIR/'train_images'/image_id
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    
    return features


# In[9]:


model = ResNet50()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

healthy = train[train['label']==4]
healthy['features'] = healthy['image_id'].progress_apply(lambda x:extract_features(x,model))


# In[10]:


features = np.array(healthy['features'].values.tolist()).reshape(-1,2048)
image_ids = np.array(healthy['image_id'].values.tolist())

# Clustering
kmeans = KMeans(n_clusters=5,n_jobs=-1, random_state=22)
kmeans.fit(features)


# In[11]:


groups = {}
for file, cluster in zip(image_ids,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)


# In[12]:


def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 25")
        start = np.random.randint(0,len(files))
        files = files[start:start+25]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(5,5,index+1);
        img = load_img(BASE_DIR/'train_images'/file)
        img = np.array(img)
        plt.imshow(img)
        plt.title(file)
        plt.axis('off')


# In[13]:


view_cluster(3)


# * We have been able to cluster most of the outliers in cluster 3 and we can easily visualize them 
# 
# * We can see that there are quite some leaves which seem to be damaged , have brown spots etc and seem not to be healthy
# 
# There are numerous dicussion threads addressing the same topic :
# * https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/198363 -- Wrong Labels
# * https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/199606 --  Quality of Labels
# 
# Now we should not be worried about the noise in the training set , but what if the noise is contained in the test set and the labelling is done similarly , then it might be a problem , we can't remove anything from training set untill we are sure
# 
# 
# So Now let's summarize this sectionn
# 
# ` Characteristics of Healthy Cassava Leaves`:
# * Mostly green in color , upright with few or no brown spots
# * A uniform texture throughout the leave be it yellow or green

# ### Learning about Disease 1 : Cassava Bacterial Blight (CBB)
# 
# Now that we know how healthy cassava leaf looks like , let's move on to learn about the first disease . First things first , `Symptoms of CBB`:
# 
# * black leaf spots and blights, angular leaf spots, and premature drying and shedding of leaves due to the wilting of young leaves
# and severe attack.
# 
# * At first, angular, water-soaked spots occur on the leaves which are restricted by the veins; the spots are more clearly seen on the lower leaf surface. The spots expand rapidly, join together, especially along the margins of the leaves, and turn brown with yellow borders (Fig. 1)
# 
# * Droplets of a creamy-white ooze occur at the centre of the spots; later, they turn yellow. 
# 
# ![](https://www.pestnet.org/fact_sheets/assets/image/cassava_bacterial_blight_173/thumbs/cassavabb_sml.jpg)
# ![](https://www.pestnet.org/fact_sheets/assets/image/cassava_bacterial_blight_173/thumbs/cassavabb2_sml.jpg)
# 
# 
# To know more visit [here](https://www.pestnet.org/fact_sheets/cassava_bacterial_blight_173.htm)

# In[14]:


plot_images(class_id=0, 
    label='CBB',
    images_number=6,verbose=1)


# * So From our knowledge of symptoms we can say that these are having CBB disease for sure and we also now know that getting the image of stem instead of the leaf itself might not be that wrong because some diseases can be judged through stem as well , so the images having stem might not be noise afterall, After viewing 6-7 different sets , there seems to be no outliers in the this category
# 
# * In some of the image like IMG - '1926670152.jpg' , the brown spot is very very small and the leaf looks more like a healthy one and a lot of healthy images also have such small brown and might be tough to identify
# 
# * From my understanding of Disease in this category , I can say RandomCropping, Contrast change , color change of any kind might not be a good idea

# ### Learning about Disease 2 : Cassava Green Mottle (CGM)
# 
# Moving onto the next disease , `Symptoms of CGM`:
# 
# * This disease causes white spotting of leaves, which increase from the initial small spots to cover the entire leaf causing
# loss of chlorophyll. Young leaves are puckered with faint to distinct yellow spots (Fig 1)
# 
# * Leaves with this disease show mottled symptoms which can be confused with symptoms of cassava mosaic disease (CMD). Severely damaged
# leaves shrink, dry out and fall off, which can cause a characteristic candle-stick appearance. (fig 2)
# 
# ![](https://www.pestnet.org/fact_sheets/assets/image/cassava_green_mottle_068/thumbs/cgmv2_sml.jpg)
# ![](https://www.pestnet.org/fact_sheets/assets/image/cassava_green_mottle_068/thumbs/cgmv_sml.jpg)
# 
# To know more visit [here](https://www.pestnet.org/fact_sheets/cassava_green_mottle_068.htm)

# In[15]:


plot_images(class_id=2, 
    label='CGM',
    images_number=12,verbose=1)


# #### Inferences
# 
# * After reading the symptoms of CGM and viewing the images from the dataset we can clearly tell the difference between CGM leaves , CBB leaves and healthy leaves
# * CGM leaves have faint to yellow spots on the leaves along the viens , CBB leaves have brown spots , and healthy leaves are either totally green or totally yellow
# * Also there are not much outliers in this class as well
# 
# 
# ### Learning about Disease 3 : Cassava mosaic disease (CMD)
# 
# `Symptoms of CMD`:
# 
# * CMD produces a variety of foliar symptoms that include mosaic, mottling, misshapen and twisted leaflets, and
# an overall reduction in size of leaves and plants
# 
# * Leaves affected by this disease have patches of normal green color mixed with different proportions of yellow and white depending on the severity

# In[16]:


plot_images(class_id=3, 
    label='CMD',
    images_number=6,verbose=1)


# ### Inferences
# 
# * We can see that CGM and CMD have very close symptoms and also have pretty similar images , often experts might get confused labelling these , we could only imagine how big a challenge it will be for the model
# 
# * There seems to be no or very less outliers in this category as well

# ### Learning about Disease 4 : Cassava Brown Streak Disease
# 
# Now the reason I have chosen this for the last , is because we have two different kind of images for this category :
# 
# * One is the Image of Leaves/plant off course
# * Another is the image of Tuburous Roots which can be easily misunderstood with potato or some kind of noise , because of the fact that dataset has noises kind of make us bias towards this assupmtion . Hence just to be clear those brown awkward looking things in the dataset are Tuburous Roots of Cassave Plant and This disease can also be identified through them
# 
# Now let's quickly look at `Symptoms of CBSD`:
# 
# * CBSD leaf symptoms consist of a characteristic yellow or necrotic vein banding which may enlarge and coalesce to form comparatively large yellow patches.
# * Tuberous root symptoms consist of dark-brown necrotic areas within the tuber and reduction in root size
# 
# So now we have a clear understanding of the two types of images present in this category and also the symptoms found in those two different images , lets look at the data

# In[17]:


plot_images(class_id=1, 
    label='CBSD',
    images_number=12,verbose=1)


# * Let's try and see if we can get the cluster of Tubular Root Images out of the data

# In[18]:


CBSD = train[train['label']==1]
CBSD['features'] = CBSD['image_id'].progress_apply(lambda x:extract_features(x,model))


# In[19]:


features_cbsd = np.array(CBSD['features'].values.tolist()).reshape(-1,2048)
image_ids_cbsd = np.array(CBSD['image_id'].values.tolist())

# Clustering
kmeans_cbsd = KMeans(n_clusters=5,n_jobs=-1, random_state=22)
kmeans_cbsd.fit(features_cbsd)


# In[20]:


groups_cbsd = {}
for file, cluster in zip(image_ids_cbsd,kmeans_cbsd.labels_):
    if cluster not in groups_cbsd.keys():
        groups_cbsd[cluster] = []
        groups_cbsd[cluster].append(file)
    else:
        groups_cbsd[cluster].append(file)


# In[21]:


def view_cluster(cluster):
    plt.figure(figsize = (25,25))
    # gets the list of filenames for a cluster
    files = groups_cbsd[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 25")
        start = np.random.randint(0,len(files))
        files = files[start:start+25]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(5,5,index+1);
        img = load_img(BASE_DIR/'train_images'/file)
        img = np.array(img)
        plt.imshow(img)
        plt.title(file)
        plt.axis('off')


# In[22]:


view_cluster(4)


# * We were successfully able to cluster the Tubular Root Images into one cluster of 80 Images and hence now we can get all the IDS and might think of Various Ideas on how to use this information

# # Summary of Our Findings : End of Step 1
# 
# Let's summarize our findings of our Initial EDA :
# 
# * Healthy Images Might not be correctly labelled , the wrongly labelled images can be found in cluster 3 of our 5 cluster.
# * Completely Yellow Leaves might not always indicate that the leaf has potential Disease
# * Brown Spots on leaves is indicative of Cassava Bacterial Blight 
# * All the Image have variety of different background and scales 
# * Images have been captured during different times of the day and thus they have different lighting and exposure
# * Cassava Green Mottle (CGM) and Cassava mosaic disease (CMD) have very similar symptoms as well as Images and might be easily mislabelled as one another . Also since there are 13k examples from Cassava Mosiac Disease , its highly likely that the most mistakes are done by model in labelling CGM as CGM
# * One Image/Cassava Plant Might contain multiple co-occurring diseases . Model would find it confusing to Label
# * CBSD have two kind of Images in the dataset , one is of the plant/leaves and the other is of roots which can be easily misunderstood as potato or some random noise , we are able to cluster all such images into cluster number 4
# 
# 
# First step in becoming a Leaf Doctor is Completed , we have successfully understood our patient and various diseases that might occur. This step will help us device unique solutions/plans to build a better solution .
# 
# <b>NOTE : I will keep on adding more such findings in this section as I keep finding more</b>

# ## Duplicates in Data : Thing we missed
# 
# After going through discussion forum I found [this](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/198202) thread which talks about the possibility of duplicate Images in the dataset. It's really interesting when we talk about duplicates in an Image dataset, because there can be two meanings to this :
# 
# * We are talking about the exact copy of an Image
# * We are talking about an Image which is similar to a particular Image . For Eg: Image 1 was cropped or rotated and stores as Image 2
# 
# Now there are several ways to find and identify Duplicate (Exact Copy) and similar Images in an Image dataset . I will use the method of Image Hashing and follow the notebook that I found [here](https://www.kaggle.com/appian/let-s-find-out-duplicate-images-with-imagehash)

# In[23]:


funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]

image_ids = []
hashes = []

for path in tqdm(glob.glob(str(BASE_DIR/'train_images'/'*.jpg' ))):
    image = Image.open(path)
    image_id = os.path.basename(path)
    image_ids.append(image_id)
    hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))


# In[24]:


hashes_all = np.array(hashes)


# Convert numpy array into torch tensor to speed up similarity calculation.

# In[25]:


hashes_all = torch.Tensor(hashes_all.astype(int)).cuda()


# Calculate similarities among all image pairs. Divide the value by 256 to normalize (0-1).

# In[26]:


get_ipython().run_line_magic('time', 'sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).cpu().numpy()/256 for i in range(hashes_all.shape[0])])')


# Thresholding

# In[27]:


indices1 = np.where(sims > 0.9)
indices2 = np.where(indices1[0] != indices1[1])
image_ids1 = [image_ids[i] for i in indices1[0][indices2]]
image_ids2 = [image_ids[i] for i in indices1[1][indices2]]
dups = {tuple(sorted([image_ids1,image_ids2])):True for image_ids1, image_ids2 in zip(image_ids1, image_ids2)}
print('found %d duplicates' % len(dups))


# Plotting the duplicate Images

# In[28]:


'''
code taken from https://www.kaggle.com/nakajima/duplicate-train-images?scriptVersionId=47295222
'''

duplicate_image_ids = sorted(list(dups))

fig, axs = plt.subplots(2, 2, figsize=(15,15))

for row in range(2):
        for col in range(2):
            img_id = duplicate_image_ids[row][col]
            img = Image.open(str(BASE_DIR/'train_images'/img_id))
            label =str(train.loc[train['image_id'] == img_id].label.values[0])
            axs[row, col].imshow(img)
            axs[row, col].set_title("image_id : "+ img_id + "  label : " + label)
            axs[row, col].axis('off')


# We have other methods to find duplicates that will help us in identifying more soft duplicates if any in the dataset , that will come in later versions of this kernel

# # Step 2 : Learning About the Methodology
# 
# Hello Doctors welcome to your second year ,in order to complete your final assignment , you now need to understand the tools you have at your disposal and how to use them , below is a step by step guide to be followed in order to learn the tools
# 
# * [Beginner Article](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
# * [Course By Andrew NG](https://www.coursera.org/learn/convolutional-neural-networks)
# * [Applying CNNS using Keras and tensorflow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow)
# * [Course from Fast.ai](https://course.fast.ai/videos/?lesson=1)

# # Step 3: Building Final Project
# 
# 
# Ohk now Docs , its time for you to build the final project . Since this is the final project it is meant for everyone to be built by themselves , here I will just write the summary of what I have used and also suggest ways that can improve the project further.
# 
# At Last I also add things to try out / look out for in the entire course of the competition
# 
# `Summary of Baseline Model`:
# 
# This model is based on the winning solution of Cassava 2019 competition and I will try to replicate it as close as possible:
# 
# * SE-ResNext50
# * Dimension = (384,384)
# * Epochs = 10
# * Custom LR scheduler 
# * Weights saved on best loss : Categorical CrossEntropy
# * Basic Augs : HorizontalFlip,VerticalFlip,Rotate,RandomBrightness,ShiftScaleRotate,cutout,centercrop,zoom,randomscale
# * No TTA
# 
# <font color ='red' >Note : As I am limited to kaggle for GPU's my five folds model is still running and hence for now I just use the pretrained weights of SeResNext50 , This notebook will be updated several times with different configs /ideas so keep tuning in</color>

# ## Configuration and utility Functions

# In[29]:


DIM = (384,384)

NUM_WORKERS = 12
TEST_BATCH_SIZE = 16
SEED = 2020

DEVICE = "cuda"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ## Augmentations

# In[30]:


def get_test_transforms():

    return albumentations.Compose(
        [albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True),
        ToTensorV2(p=1.0)
        ]
    )


# # Cassava Dataset

# In[31]:


class CassavaDataset(Dataset):
    def __init__(self,image_ids,labels,dimension=None,augmentations=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.dim = dimension
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self,idx):
        
        img = cv2.imread(str(BASE_DIR/'test_images'/self.image_ids[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                         
        if self.dim:
            img = cv2.resize(img,self.dim)
        
        if self.augmentations:
            augmented = self.augmentations(image=img)
            image = augmented['image']
                         
        return {
            'image': image,
            'target': torch.tensor(self.labels[idx],dtype=torch.float)
        }


# # Model : SE_Resnext50

# In[32]:


class CassavaModel(nn.Module):
    def __init__(self, model_name='seresnext50_32x4d',out_features=5,pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        n_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Linear(n_features, out_features)

    def forward(self, x):
        x = self.model(x)
        return x


# # Prediction Function Single Model

# In[33]:


def predict_single_model(data_loader,model,device):
    model.eval()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))
    fin_out = []
    
    with torch.no_grad():
        
        for bi, d in tk0:
            images = d['image']
            targets = d['target']
            
            images = images.to(device)
            targets = targets.to(device)
            
            batch_size = images.shape[0]
            
            outputs = model(images)
            
            fin_out.append(F.softmax(outputs, dim=1).detach().cpu().numpy())
            
    return np.concatenate(fin_out)


# # Engine

# In[34]:


sample_sub = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')


# In[35]:


def predict(weights):
    '''
    weights : List of paths in case of K fold model inference
    '''
    pred = np.zeros((len(sample_sub),5,5))
    
    # Defining DataSet
    test_dataset = CassavaDataset(
        image_ids=sample_sub['image_id'].values,
        labels=sample_sub['label'].values,
        augmentations=get_test_transforms(),
        dimension = DIM
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    
    # Defining Device
    device = torch.device("cpu")
    
    for i,weight in enumerate(weights):
        # Defining Model for specific fold
        model = CassavaModel(out_features=5,pretrained=True)
        
        # loading weights
        #model.load_state_dict(torch.load(weight))
        model.to(device)
        
        #predicting
        pred[:,:,i] = predict_single_model(test_loader,model,device)
    
    return pred


# # Preparing Final Submission

# In[36]:


pred = predict([1])
print(pred)


# In[37]:


pred = pred.mean(axis=-1)
print('Prediction Before Argmax',pred)
pred = pred.argmax(axis=1)
print('Final Prediction',pred)


# In[38]:


sample_sub['label'] = pred
sample_sub.head()


# In[39]:


sample_sub.to_csv('submission.csv',index=False)


# # Conclusion
# 
# There is a lot to try as the competition is just starting , I will try to keep this notebook updated
# 
# Thanks for reading my notebook , I hope you got something helpful out of it
