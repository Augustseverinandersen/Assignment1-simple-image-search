# System tools
import os
import sys
sys.path.append("utils")
import argparse


# data munging
import numpy as np
import pandas as pd
from numpy.linalg import norm
from tqdm import tqdm # Creates progress bars

# tensorflow
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16)
                                                # preprocess_input)


# Scikit learn
from sklearn.neighbors import NearestNeighbors

# Helper function

import helper_functions as hf





def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments // "--name" is what you feed it in the command line
    parser.add_argument("--choose_image", type=int, default = 28, help = "Chose a number from the image path list")
    # parse the arguments from command line
    args = parser.parse_args()
    return args


# Loading pretrained model VGG16
def pretrained_model():
    print("Loading pretrained model VGG16")
    model = VGG16(weights='imagenet', # Weights that were trained on imagenet 
                include_top=False, # dont include classifiers.
                pooling='avg', # Average pooling
                input_shape=(224, 224, 3)) # Input shape. 224 by 224 pixel size, and three color channels
    return model


def path_to_images(): # Joining the validation and train images together. There is no need for validation images in this script
    print("Creating image path list")
    train_dir = os.path.join("data","17_flowers","train")
    validation_dir = os.path.join("data","17_flowers","validation")
    
    filenames = [] # Empty list to store all the filepaths
    print("Appending filepaths for train images")
    for root, dirs, files in os.walk(train_dir): # For entire file path for the train images append to empty list.
        for file in files:
            if file.endswith(".jpg"): # If they end on .jpg
                filepath = os.path.join(root, file)
                filenames.append(filepath)

    print("Appending filepath for validation images")        
    for root, dirs, files in os.walk(validation_dir): # For entire file path for the validation images append to empty list
        for file in files:
            if file.endswith(".jpg"):
                filepath = os.path.join(root, file)
                filenames.append(filepath)
                
    filenames = sorted(filenames, key=lambda x: x.split("/")[-1]) # Sorting the list of filenames, based on the last part of the path
    
    return filenames


# Extracting features for each image
def feature_extraction(filenames, model):
    print("Feature extraction")
    feature_list = []
    for i in tqdm(range(len(filenames)), position = 0, leave = True): # for every image in the filenames (images). 
        # with tqdm set a progress bar 
        feature_list.append(hf.extract_features(filenames[i], model)) # Using helper function. Returns a normalized feature vector for each image.
    return feature_list


# Nearest neighbours
def nearest_neighbours(feature_list): 
    print("Nearest neighbours ")   
    neighbors = NearestNeighbors(n_neighbors=10, # find 10 closet 
                                algorithm='brute', # use bruteforce algoritm 
                                metric='cosine').fit(feature_list) # use cosine  
                                # fit it to feature list.
    return neighbors


# Calculating nearest neighbour for target
def calc_nearest_neighbour(args, neighbors, feature_list):
    print("Calculating nearest neighbours")
    chosen_image = [feature_list[args.choose_image]]
    distances, indices = neighbors.kneighbors(chosen_image)
    # first output  = cosine similarity 
    # Second output = indicies of the images. 
    return distances, indices


# Save indices
def five_indices(distances, indices):
    print("Finding five nearest neighbours")
    idxs = [] # six closet images. including chosen image
    for i in range(1,6):
        print(distances[0][i], indices[0][i])
        idxs.append(indices[0][i]) # Append the images indice to the empty list
    print(idxs)
    return idxs


# Getting filepath for five closet images
def filepath_indices(filenames, idxs):
    print("Getting filename and filepath for the nearest neighbours.")
    img_name = [] # Empty list to contain filename
    for idx in idxs: # for i in the list idxs
        img_name.append(filenames[idx]) # Get the file name of the corresponding idx
    
    return img_name 



def creating_dataframe(img_name, distances, filenames, args):
    # Create a dictionary key filepath containing list filepaths, and key distance containg nearest neighbor distance
    five_closest_dictionary = {'Filepath': img_name, 'Distance': distances[0][1:6]} #  including chosen image
    # Using pandas to create dataframe 
    five_closest_dataframe = pd.DataFrame(five_closest_dictionary)
    outpath = os.path.join("out", "nearest_neighbor_five_images.csv") # defining output path
    five_closest_dataframe.to_csv(outpath, index= False) # transforming my dataframe to a csv file, with no index.
    print("Path to your chosen image: " + filenames[args.choose_image]) # Printing path to chosne image

def main_function():
    args = input_parse()
    model = pretrained_model()
    filenames = path_to_images()
    feature_list = feature_extraction(filenames, model)
    neighbors = nearest_neighbours(feature_list)
    distances, indices= calc_nearest_neighbour(args, neighbors, feature_list)
    idxs = five_indices(distances, indices)
    img_name = filepath_indices(filenames, idxs)
    creating_dataframe(img_name, distances, filenames, args)


if __name__ == "__main__":
    main_function()
