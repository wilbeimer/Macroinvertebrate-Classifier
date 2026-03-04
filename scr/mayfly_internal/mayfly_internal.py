import rawpy
import imageio
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.keras import layers, models
import itertools
from tqdm import tqdm
import pandas as pd
import joblib
import PIL.Image as Image
from pathlib import Path
import random
from sklearn.metrics import roc_curve, auc

### THESE FUNCTIONS ARE NOT TO BE CALLED WHEN TRAINING MODELS ###

#   - Call the functions in mayflylib.py

def NB_pred_single(image, model, slice_factor):
    subimages = to_chunks(image, slice_factor)
    predictions = NB_predict_on_images(model,subimages)
    return (subimages, predictions, (len(vertical_slices)-1, len(horizontal_slices) -1))

def to_chunks(image, slice_factor):
    sliced_image = image[::slice_factor,::slice_factor]
    if(len(image.shape) != 3):
        raise TypeError(f"Wrong shape of array!!! You have {image.shape}")
    vertical_slices, horizontal_slices = get_slices(sliced_image,224)
    product = np.array(list(itertools.product(vertical_slices[:-1], horizontal_slices[:-1])))
    slices = np.empty(len(product), dtype=object)
    slices[:] = list(map(tuple, product))
    subimages = np.array([sliced_image[chunk] for chunk in slices])
    return subimages, (len(vertical_slices),len(horizontal_slices))

## Sumarry functions for the Naive Bayes Model
def pre_summary(long,k):
    central_moments = np.array(list(map(lambda n :np.mean((long - np.mean(long))**n)/(np.std(long)**n),range(2,k+1))))
    raw_moments = np.array(list(map(lambda n :np.mean(long**n)/(np.std(long)**n),range(1,k+1))))
    extreme = np.array([max(long)-min(long),max(long)-raw_moments[0],min(long)-raw_moments[0]])
    summary = np.concatenate((central_moments,raw_moments,extreme),axis = 0)
    
    return(summary)

def image_summary(image, k = 3):
    pixels = image.flatten().tolist() # flattens the image into a long list
    reds   = np.array(pixels[0::3])
    greens = np.array(pixels[1::3]) # seperates each color into its own list
    blues  = np.array(pixels[2::3])
    red_summary = pre_summary(reds, k)
    blue_summary = pre_summary(blues, k)
    green_summary = pre_summary(greens, k)
    summary = np.concatenate((red_summary,blue_summary,green_summary),axis = 0) # returns the 3*k moments
    
    return(summary)

## Instead of getting the summary of images, just pipe the images directly into is function and get predictions
def NB_predict_on_images(model, images, M=3):
 # number of raw moments to use as predictors

    summarylist = np.array([image_summary(x, k = M) for x in images])
    predictions = model.predict(summarylist)
    
    return(predictions)

def get_slices(rgb,size):
    # DEFINES THE SUBSETS
    height = rgb.shape[0]
    width = rgb.shape[1]

    vertical_slices = []
    horizontal_slices = []

    K = height//size
    for k in range(K):
        t = slice(k*size,min([(k+1)*size,height-1]))
        vertical_slices.append(t)

    N = width//size
    for n in range(N):
        t = slice(n*size,min([(n+1)*size,width-1]))
        horizontal_slices.append(t)
        
    return (vertical_slices,horizontal_slices)


    
def zoom_out(FullImage, horizontal_slice, vertical_slice, sizex):
    changer = (size-224)//2 #This is the amount we'll subtract from the slice starts and add to slice ends.
    original_x = horizontal_slice
    original_y = vertical_slice

    NewYStart = int(original_y.start - changer)
    NewYEnd = int(original_y.stop + changer)
    NewXStart = int(original_x.start - changer)
    NewXEnd = int(original_x.stop + changer)

    #These if statements are used to shift the slices if they go off the edges. PLEASE NOTE THAT OUR IMAGES HAVE SHAPE (6744, 4502, 3)

    if NewYStart < 0:
        NewYEnd -= NewYStart
        NewYStart = 0

    if NewYEnd > 6744:
        NewYStart -= (NewYEnd - 6744)
        NewYEnd = 6744

    if NewXStart < 0:
        NewXEnd -=NewXStart
        NewXStart = 0

    if NewXEnd > 4502:
        NewXStart -= (NewXEnd - 4502)
        NewXEnd = 4502

    ScaledXSlice = __builtins__.slice(NewXStart, NewXEnd)
    ScaledYSlice = __builtins__.slice(NewYStart, NewYEnd)

    return(ScaledXSlice, ScaledYSlice)

def grey_if_true(image, grey):
    if not grey:
        return image
    elif grey:
        grey_single = image.mean(axis=2, keepdims=True)
        grey_3channel = np.repeat(grey_single, 3, axis=2).astype(image.dtype)
        return grey_3channel

