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
import mayfly_internal as mai

def hand_labeling(): 
    pass

def write_non_empty():
    pass

def nb_assisted_hand_labeling():
    pass

def image_summary():
    pass

def NB_predictions(images, model,slice_factor):
    subimage_list = []
    prediction_list = []
    
    for image in tqdm(images, desc = "convoluting transpositions"):
            
        subimages, predictions, shape = mai.NB_pred_single(image,model,slice_factor)
        subimage_list.append(subimages)
        prediction_list.append(predictions)
    subimages = np.stack(subimage_list, axis = 0)
    predictions = np.stack(prediction_list, axis = 0)
    return subimages, predictions, shape

### HAND LABELING IMAGES FUNCTION
def label_imageset(images,predictions,shape):
    ## Get a list of images to review
    ## Get the original indexes of the images so that adjusted labels can be updated.

    photo_indexed_review_list = [images[i][np.where(predictions[i] == 1)] for i in range(images.shape[0])]
    review_list = np.concatenate(photo_indexed_review_list, axis = 0)
    original_indexes = []
    for k in range(images.shape[0]):
        for i in np.where(predictions[k] == 1)[0]:
            original_indexes.append((k,i))
            
    

    updated_labels = []
    i = 0
    while i < len(review_list):

        # show the image
        clear_output(wait=True)
        plt.clf()
        plt.axis("off")
        plt.imshow(review_list[i])
        plt.show()

        # ask user to label 
        print(f"Progress: ({len(updated_labels)}/{review_list.shape[0]})")
        user_inp = input("Is this empty(y/n)?")

        if user_inp == "c":
            break
        elif user_inp in ["y","n"]:
            if user_inp == "y":
                updated_labels.append(0)
            elif user_inp == "n":
                updated_labels.append(1)
            i+=1

        elif user_inp == "b":
            i-=1
            updated_labels.pop()
            continue
            clear_output(wait=True)
            plt.clf()
            plt.axis("off")
            plt.imshow(review_list[i])
            plt.show()

    # updates all with the humans labels
    updated_labels = np.array(updated_labels)
    for i in range(updated_labels.shape[0]):
        predictions[original_indexes[i]] = updated_labels[i]

        
        
    # the viewer can review their labels
    label_quality = []
    for j in range(predictions.shape[0]):
        # show the image

        view = visualize_predictions(images[j], predictions[j], shape)

        clear_output(wait=True)
        plt.clf()
        plt.axis("off")
        plt.imshow(view)
        plt.show()

        while True:
            quality = input(f"Should this be relabelled?(y/n)   {j}/{predictions.shape[0]}")
            if quality == "n":
                label_quality.append(0)  ## 0 means /abeling is not necessary
                break
            elif quality == "y":
                label_quality.append(1)  ## 1 means relabeling is necessary
                break
    label_quality = np.array(label_quality)
    
    low_quality_indexes = np.where(label_quality == 1)
    if low_quality_indexes[0].shape[0]>0:
        relabeled = label_imageset(images[low_quality_indexes],predictions[low_quality_indexes],shape)
        predictions[low_quality_indexes] = relabeled
    return predictions 


def visualize_predictions(subimages, predictions,shape):
    subimages = subimages.reshape(shape[0], shape[1],224,224,3)
    predictions = predictions.reshape(shape[0], shape[1])
    k = 0
    while k<(shape[0]):
        n = 0
        while n<(shape[1]):
            subimages[k,n] = mai.grey_if_true(subimages[k,n],not bool(predictions[k,n]))
            n+=1
        k+=1
    x, y, h, w, c = subimages.shape
    full = subimages.transpose(0, 2, 1, 3, 4).reshape(x * h, y * w, c)
    return full
