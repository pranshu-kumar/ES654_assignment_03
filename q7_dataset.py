# Reference: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

from fastbook import *
from PIL import Image
import os

import sys
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
import random as rand
import numpy as np



# Download Dataset
def download_grizzly_bears(num_images):
    urls = search_images_ddg('grizzly bear', max_images=num_images)

    for i in range(len(urls)):
        download_url(urls[i], 'images/bear' + str(i))

def download_panthers(num_images):
    urls = search_images_ddg('black panther animal', max_images=num_images)

    for i in range(len(urls)):
        download_url(urls[i], 'images/panther' + str(i+41) + '.jpeg')

# download_panthers(2)

# rename files
def rename_files():
    for count, filename in enumerate(os.listdir("images/")):
            dst = 'bear' + str(count) +  ".jpeg"
            src ='images/'+ filename
            dst ='images/'+ dst

            os.rename(src, dst)

# Resize Images
def resize_images(path):
    for file in os.listdir(path):
        f_img = path + file
        img = Image.open(f_img)
        img = img.resize((200,200))
        img.save(f_img)
        
# rename_files()
# resize_images('images/')

# Create Train and Test Subdirectories
dataset_home = 'images/'
def create_subdirs():
    dataset_home = 'images/'
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ['bears/', 'panthers/']
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)

# create_subdirs()

def copy_train_imgs():
    src_dir = 'images/'

    for file in listdir(src_dir):
        if file == 'test' or file == 'train':
            continue  
        src = src_dir + file
        dst_dir = 'train/'

        dst = dataset_home + dst_dir + file
        copyfile(src, dst)
        
# copy_train_imgs()

# seed random number generator
def copy_test_imgs(): 
    seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.25
    # copy training dataset images into subdirectories
    src_directory = 'train/'
    for file in listdir(dataset_home + src_directory):
        if file == 'bears' or file == 'panthers':
            continue
        src = dataset_home + src_directory + file  # images/train/file
        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'test/'
        if file.startswith('bear'):
            dst = dataset_home + dst_dir + 'bears/'  + file
            copyfile(src, dst)
        elif file.startswith('panther'):
            dst = dataset_home + dst_dir + 'panthers/'  + file
            copyfile(src, dst)


# copy_test_imgs()




