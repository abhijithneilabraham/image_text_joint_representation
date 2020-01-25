#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:16:28 2020

@author: abhijithneilabraham
"""

import numpy as np
features = np.load("resnet50-features.10k.npy")
print(features.shape)
import tensorflow as tf
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input
resnet_model = ResNet50(weights='imagenet',include_top=False)
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = resnet_model.predict(x)
    return np.expand_dims(features.flatten(), axis=0)
features = extract_features('test.jpg')
print(features)
