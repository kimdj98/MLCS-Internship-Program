# -*- coding: utf-8 -*-
"""data_loader.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sp2ImDqLA8Gk2UO-Rke3Dnaxp3GbykR3

# Basic Python Programming 004 Custom Dataset
"""

import os, glob
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
from numpy import asarray

"""#### custom dataset for image classification
Chest X-RAY-images (Pneumonia classification) (dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
"""

from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union

class DataLoader():
    def __init__(self, PATH,
                 batch_size:Optional[int]=64,
                 normalize:Optional[bool]=False,
                 shuffle:Optional[bool]=False,
                 get_statistics:Optional[bool]=False, # to get mean and standard deviation of datas
                 mu:Optional[float]=0.0,
                 sd:Optional[float]=0.0,
                 ):

        # initialize data
        self.batch_images = np.zeros((batch_size,900,1200))
        self.batch_labels = np.zeros(batch_size)

        # get NORMAL and PNEUMONIA folder path
        self.NORMAL_PATH = os.path.join(PATH, "NORMAL")
        self.PNEUMONIA_PATH = os.path.join(PATH, "PNEUMONIA")
        # get paths for normal images and pneumonia images
        self.NORMAL_PATHS = glob.glob(os.path.join(self.NORMAL_PATH, "*.jpeg"))
        self.PNEUMONIA_PATHS = glob.glob(os.path.join(self.PNEUMONIA_PATH, "*.jpeg"))
        # concatenate images path and make label datas 
        self.image_paths = self.NORMAL_PATHS + self.PNEUMONIA_PATHS
        self.labels = [0] * len(self.NORMAL_PATHS) + [1] * len(self.PNEUMONIA_PATHS)
        
        # zip image_paths and labels for shuffling
        self.datas = list(zip(self.image_paths, self.labels))
        if shuffle:
            random.shuffle(self.datas)
        self.image_paths, self.labels = zip(*self.datas) # shuffled image paths and labels
        
        # convert image to array with batch_size
        self.normalize_flag = normalize
        self.mu = mu
        self.sd = sd
        for i in range(batch_size):
            image = Image.open(self.image_paths[i]).convert("L") # open image in grayscale
            image = image.resize((1200,900))
            image = np.asarray(image, dtype=np.float32)
            image = self.normalize(image, mu, sd)
            self.batch_images[i] = image
            self.batch_labels[i] = self.labels[i]
        
        # save batch of numpy images to npz file
        DATASET_PATH = "./datasets/batch_data.npz"
        np.savez(DATASET_PATH,
                batch_images = self.batch_images,
                batch_labels = self.batch_labels)

        # code to get mean and standard deviation of dataset        
        if get_statistics:
            self.mu = 0.0
            self.mu_squared = 0.0
            self.sd = 0.0

            for i, (image_path, label) in enumerate(self.datas):
                image = Image.open(image_path)
                image = image.resize((1200,900))
                image = np.asarray(image, dtype=np.float32)
                self.mu += np.sum(image)
                self.mu_squared += np.sum(np.square(image))

            self.mu /= (i+1)*1200*900
            self.mu_squared /= (i+1)*1200*900

            self.sd = np.sqrt(self.mu_squared - self.mu * self.mu)
            print(f"mean: {self.mu:.4f}")
            print(f"standard deviation: {self.sd:.4f}")

    def normalize(self, arr, mu, sd):
        if self.normalize_flag:
            arr -= mu
            arr /= sd
        return arr

    def load(self):
        d = np.load("./datasets/batch_data.npz")
        self.batch_images = d["batch_images"]
        self.batch_labels = d["batch_labels"]

data_loader = DataLoader(PATH="chest_xray/train", normalize=True, shuffle=True, mu=134.11, sd=52.92)

"""Check shuffle works well"""

print(data_loader.batch_labels)

"""Check if data loader size is correct"""

print(data_loader.batch_images.shape)
print(data_loader.batch_labels.shape)

"""Check data loader load correctly"""

data_loader.load()

print(data_loader.batch_images.shape)
print(data_loader.batch_labels.shape)

"""Get statistics(mean and standard deviation)"""

train_loader = DataLoader(PATH="chest_xray/train", get_statistics=True)

"""cf) debugging"""

print(len(train_loader.NORMAL_PATHS))
print(len(train_loader.PNEUMONIA_PATHS))
print(len(train_loader.labels))