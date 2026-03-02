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
def NB_predict_on_images(model, images, M = 3):
 # number of raw moments to use as predictors

    summarylist = np.array([image_summary(x, k = M) for x in images])
    predictions = model.predict(summarylist)
    
    return(predictions)

def NB_preds(images, model,slice_factor):
    subimage_list = []
    prediction_list = []
    
    for image in tqdm(images, desc = "convoluting transpositions"):
            
        subimages, predictions, shape = NB_pred_single(image,model,slice_factor)
        subimage_list.append(subimages)
        prediction_list.append(predictions)
    subimages = np.stack(subimage_list, axis = 0)
    predictions = np.stack(prediction_list, axis = 0)
    return subimages, predictions, shape
        
def NB_pred_single(image, model, slice_factor):
    sliced_image = image[::slice_factor,::slice_factor]
    if(len(image.shape) != 3):
        raise TypeError(f"Wrong shape of array!!! You have {image.shape}")
    vertical_slices, horizontal_slices = get_slices(sliced_image,224)
    product = np.array(list(itertools.product(vertical_slices[:-1], horizontal_slices[:-1])))
    slices = np.empty(len(product), dtype=object)
    slices[:] = list(map(tuple, product))
    subimages = np.array([sliced_image[chunk] for chunk in slices])
    predictions = NB_predict_on_images(model,subimages)
    return (subimages, predictions, (len(vertical_slices)-1, len(horizontal_slices) -1))



###### Defines the structure of the CNN that is used for relabeling

# defining the structure of the CNN
def nothing_reclassifier_cnn(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def make_roc_curve(CNN_model, X_test, y_test):
    y_probs = CNN_model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # The "Random Guess" line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    print(data['images'].shape)

    
def visualize_predictions(subimages, predictions,shape):
    subimages = subimages.reshape(shape[0], shape[1],224,224,3)
    predictions = predictions.reshape(shape[0], shape[1])
    k = 0
    while k<(shape[0]):
        n = 0
        while n<(shape[1]):
            subimages[k,n] = grey_if_true(subimages[k,n],not bool(predictions[k,n]))
            n+=1
        k+=1
    x, y, h, w, c = subimages.shape
    full = subimages.transpose(0, 2, 1, 3, 4).reshape(x * h, y * w, c)
    return full
    
## Gets the slices of the original image and returns them
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

## grey is boolean value, 
## just returns a grey version of the image that is the same shape np.array
def grey_if_true(image, grey):
    if not grey:
        return image
    elif grey:
        grey_single = image.mean(axis=2, keepdims=True)
        grey_3channel = np.repeat(grey_single, 3, axis=2).astype(image.dtype)
        return grey_3channel

    
    
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

        view = rel.visualize_predictions(images[j], predictions[j], shape)

        clear_output(wait=True)
        plt.clf()
        plt.axis("off")
        plt.imshow(view)
        plt.show()

        while True:
            quality = input(f"Should this be relabelled?(y/n)   {j}/{predictions.shape[0]}")
            if quality == "n":
                label_quality.append(0)  ## 0 means relabeling is not necessary
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