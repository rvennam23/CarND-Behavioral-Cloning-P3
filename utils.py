# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:35:08 2017

@author: rvennam
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def drop_low_steeringangle_data(data):
    """ Drop some low steering angle samples  """
    index = data[abs(data['steering'])<.05].index.tolist()
    rows = [i for i in index if np.random.randint(10) < 4]
    data = data.drop(data.index[rows])
    print("Dropped %s rows with low steering"%(len(rows)))
    return data

def preprocess_img(img):
    """Returns croppped image
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return img[60:135, : , ]

    
def process_img_from_path(img_path):
    """Returns Croppped Image
    for given img path.
    """
    return preprocess_img(plt.imread(img_path))


def get_batch(data, batch_size):
    """Returns randomly sampled data
    from given pandas df  .  
    """
    return data.sample(n=batch_size)


def choose_image(data, value, data_path):
    """ Returns randomly selected right, left or center images
    and their corrsponding steering angle.
    The probability to select center is twice of right or left. 
    """ 
    random = np.random.randint(4)
    if (random == 0):
        img_path = data['left'][value].strip()
        shift_ang = .20
    if (random == 1 or random == 3):
        img_path = data['center'][value].strip()
        shift_ang = 0.
    if (random == 2):
        img_path = data['right'][value].strip()
        shift_ang = -.20
    img = process_img_from_path(os.path.join(data_path, img_path))
    steer_ang = float(data['steering'][value]) + shift_ang
    return img, steer_ang


def trans_image(image, steer):
    """ Returns translated image and 
    corrsponding steering angle.
    """
    trans_range = 100
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (320,75))
    return image_tr, steer_ang

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_flip_image(image,steering_angle):  
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle



def training_image_generator(data, batch_size, data_path):
    """
    Train data generator
    """
    while 1:
        batch = get_batch(data, batch_size)
        features = np.empty([batch_size, 75, 320, 3])
        labels = np.empty([batch_size, 1])
        for i, value in enumerate(batch.index.values):
            # Randomly select right, center or left image
            img, steer_ang = choose_image(data, value, data_path)
            img = img.reshape(img.shape[0], img.shape[1], 3)          
            # Random Translation Jitter
            img, steer_ang = trans_image(img, steer_ang)
            # Add Random Brightness to image
            img = random_brightness(img)
            # Randomly Flip Images
            img, steer_ang = random_flip_image(img, steer_ang)
            features[i] = img
            labels[i] = steer_ang
        yield np.array(features), np.array(labels)
        

def get_images(data, data_path):
    """
    Validation Generator
    """
    while 1:
        for i in range(len(data)):
            img_path = data['center'][i].strip()
            img = process_img_from_path(os.path.join(data_path, img_path))
            img = img.reshape(1, img.shape[0], img.shape[1], 3)
            steer_ang = data['steering'][i]
            steer_ang = np.array([[steer_ang]])
            yield img, steer_ang