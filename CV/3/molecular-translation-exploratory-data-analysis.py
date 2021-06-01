#!/usr/bin/env python
# coding: utf-8

# # Bristol-Myers Squibb – Molecular Translation - Exploratory Data Analysis
# 
# Quick Exploratory Data Analysis for [Bristol-Myers Squibb – Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation) challenge    
# 
# In this competition, you are provided with images of chemicals, with the objective of predicting the corresponding International Chemical Identifier (InChI) text string of the image. The images provided (both in the training data as well as the test data) may be rotated to different angles, be at various resolutions, and have different noise levels.

# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/22422/logos/header.png)

# <a id="top"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:black; border:0' role="tab" aria-controls="home"><center>Quick Navigation</center></h3>
# 
# * [Overview](#1)
# * [Data Visualization](#2)
#     
# 
# * [Competition Metric](#100)
# * [Submission](#101)

# <a id="1"></a>
# <h2 style='background:black; border:0; color:white'><center>Overview<center><h2>

# In[ ]:


import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt


# **train_labels.csv** - ground truth InChi labels for the training images

# In[ ]:


TRAIN_LABELS_PATH = "../input/bms-molecular-translation/train_labels.csv"

df_train_labels = pd.read_csv(TRAIN_LABELS_PATH, index_col=0)
df_train_labels


# **train/** - the training images, arranged in a 3-level folder structure by image_id

# In[ ]:


def convert_image_id_2_path(image_id: str) -> str:
    return "../input/bms-molecular-translation/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id 
    )


# In[ ]:


def visualize_train_batch(image_ids, labels):
    plt.figure(figsize=(16, 12))
    
    for ind, (image_id, label) in enumerate(zip(image_ids, labels)):
        plt.subplot(3, 3, ind + 1)
        image = cv2.imread(convert_image_id_2_path(image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
#         print(f"{ind}: {label}")
        plt.title(f"{label[:30]}...", fontsize=10)
        plt.axis("off")
    
    plt.show()


# In[ ]:


def visualize_train_image(image_id, label):
    plt.figure(figsize=(10, 8))
    
    image = cv2.imread(convert_image_id_2_path(image_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(f"{label}", fontsize=14)
    plt.axis("off")
    
    plt.show()


# <a id="2"></a>
# <h2 style='background:black; border:0; color:white'><center>Data Visualization<center><h2>

# In[ ]:


sample_row = df_train_labels.sample(5)
for i in range(5):
    visualize_train_image(
        sample_row.index[i], sample_row["InChI"][i]
    )


# In[ ]:


tmp_df = df_train_labels[:9]
image_ids = tmp_df.index
labels = tmp_df["InChI"].values

visualize_train_batch(image_ids, labels)


# In[ ]:


tmp_df = df_train_labels.sample(9)
image_ids = tmp_df.index
labels = tmp_df["InChI"].values

visualize_train_batch(image_ids, labels)


# In[ ]:


def visualize_batch_without_labels(image_ids):
    plt.figure(figsize=(16, 16))
    
    for ind, image_id in enumerate(image_ids):
        plt.subplot(5, 5, ind + 1)
        image = cv2.imread(convert_image_id_2_path(image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.axis("off")
    
    plt.show()


# In[ ]:


tmp_df = df_train_labels.sample(25)
image_ids = tmp_df.index

visualize_batch_without_labels(image_ids)


# In[ ]:


tmp_df = df_train_labels.sample(1000)
h_shape = []
w_shape = []
for image_id in tmp_df.index:
    image = cv2.imread(convert_image_id_2_path(image_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_shape.append(image.shape[0])
    w_shape.append(image.shape[1])


# In[ ]:


plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
plt.hist(np.array(h_shape) * np.array(w_shape), bins=50)
plt.xticks(rotation=45)
plt.title("Area Image Distribution", fontsize=14)
plt.subplot(1, 3, 2)
plt.hist(h_shape, bins=50)
plt.title("Height Image Distribution", fontsize=14)
plt.subplot(1, 3, 3)
plt.hist(w_shape, bins=50)
plt.title("Width Image Distribution", fontsize=14);


# **test/** - the test images, arranged in the same folder structure as train/

# <a id="100"></a>
# <h2 style='background:black; border:0; color:white'><center>Competition Metric<center><h2>

# Submissions are evaluated on the mean [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) between the InChi strings you submit and the ground truth InChi values.

# The Levenshtein distance between two strings **a,b** (of length **|a|** and **|b|** respectively) is given by **lev(a,b)** where

# $${\displaystyle \qquad \operatorname {lev} (a,b)={\begin{cases}|a|&{\text{ if }}|b|=0,\\|b|&{\text{ if }}|a|=0,\\\operatorname {lev} (\operatorname {tail} (a),\operatorname {tail} (b))&{\text{ if }}a[0]=b[0]\\1+\min {\begin{cases}\operatorname {lev} (\operatorname {tail} (a),b)\\\operatorname {lev} (a,\operatorname {tail} (b))\\\operatorname {lev} (\operatorname {tail} (a),\operatorname {tail} (b))\\\end{cases}}&{\text{ otherwise.}}\end{cases}}}$$

# where the **tail** of some string **x** is a string of all but the first character of **x**, and **x[n]** is the **n**th character of the string **x**, starting with character 0.
# 
# Note that the first element in the minimum corresponds to deletion (from **a** to **b**), the second to insertion and the third to replacement.

# <img style="height:600px" src="https://miro.medium.com/max/554/1*bEWdxv_FoTQurG9fyS3nSA.jpeg">
# <cite>The image from <a href="https://medium.com/@ethannam/understanding-the-levenshtein-distance-equation-for-beginners-c4285a5604f0">Understanding the Levenshtein Distance Equation for Beginners</a></cite>

# <a id="101"></a>
# <h2 style='background:black; border:0; color:white'><center>Submission<center><h2>

# **sample_submission.csv** - a sample submission file in the correct format

# In[ ]:


SAMPLE_SUBMISSION_PATH = "../input/bms-molecular-translation/sample_submission.csv"

df_sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH, index_col=0)
df_sample_submission


# In[ ]:


df_sample_submission.to_csv("submission.csv")


# In[ ]:




