"""

Functions that will be used in the script *histogram_image_search_alogrithm.py

"""
# Data munging
import cv2
import pandas as pd
import numpy as np
from numpy.linalg import norm

# TensorFlow
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)



def image_processing_histogram(images):
    # creating histogram for the img
    flowers_hist = cv2.calcHist([images], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256]) # Using cv2s function "calcHist", to get histogram for an image
    # cv2.calcHist arguments: takes the an image, colour channels, no making, the size of the histogram, pixel range for each color.                       
    return flowers_hist


def image_processing_normalization(images):
    # Normalising the img
    flowers_histnorm = cv2.normalize(images, images, 0, 1.0, cv2.NORM_MINMAX) # Using cv2s normalize function to normalise an image
    # cv2.normalize arguments: input and out image (the same), 0-1 rescaling range of pixels, the normalization method.
    return flowers_histnorm


def comparison(image, images):
    # Comparing the img score with my chosen image
    comparing_score = round(cv2.compareHist(image, images, cv2.HISTCMP_CHISQR), 2) # Using cv2s function compareHist to compare the histograms 
    #cs2.compareHist arguments: an image to be compared with chosen_image, using chi-squared distance metric, rounding output to two decimals
    return comparing_score


def filename_score(empty_list, file_name, score):
    # Saving the filename and the score, in my empty list as a tuple.
    empty_list.append((file_name, score))


""" 

Function to be used in the script *feature_extraction_image_search_algorithm.py* 
Created by Ross

"""



# Helper function
def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)
    """
    # Define input image shape - remember we need to reshape
    input_shape = (224, 224, 3)
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image 
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features) # getting norm from numpy 
  
    return normalized_features

