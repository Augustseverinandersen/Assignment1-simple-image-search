# base tools
import os, sys
sys.path.append(os.path.join(".."))

# data analysis
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm # Create progress bars for for loops, like tensor flow does.

# tensorflow
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from sklearn.neighbors import NearestNeighbors
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append(".")

import argparse
import pandas as pd

def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments // "--name" is what you feed it in the command line
    parser.add_argument("--choose_image", type=int, default = 250)
    parser.add_argument("--directory", type=str )
    # parse the arguments from command line
    args = parser.parse_args()
    return args


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
    # preprocess image - see last week's notebook
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features) # getting norm from numpy 
  
    return normalized_features





# Loading pretrained model VGG16
def pretrained_model():
    model = VGG16(weights='imagenet', 
                include_top=False, # dont include classifiers.
                pooling='avg',
                input_shape=(224, 224, 3))
    return model

# Iterating over folder containing all images 
# path to the datasets

def files_path(args):
    print("Find path to all images") 
    root_dir = args.directory
    filenames = [root_dir + "/" + name for name in sorted(os.listdir(root_dir)) if name.endswith(".jpg")] # list comprehension.
    # For every file in root_dir, get the list of all image names, sort them, combine them with root dir.
    return filenames

# Extracting features for each image

def feature_extraction(filenames, model):
    feature_list = []
    for i in tqdm(range(len(filenames)), position = 0, leave = True): # for every image in the filenames (images). 
        # with tqdm set a progress bar 
        # REmove notebook - position - leave
        feature_list.append(extract_features(filenames[i], model))
    return feature_list


# Nearest neighbours

def nearest_neighbours(feature_list):    
    neighbors = NearestNeighbors(n_neighbors=10, # find 10 closet 
                                algorithm='brute', # use bruteforce algoritm 
                                metric='cosine').fit(feature_list) # use cosine  
                                #fit it to feature list.
    return neighbors


# Calculating nearest neighbour for target
# MAKE THIS AN ARG PARSE 
def calc_nearest_neighbour(args, neighbors, feature_list):
    chosen_image = [feature_list[args.choose_image]]
    distances, indices = neighbors.kneighbors(chosen_image) # on image 250. 
    # first output  = cosine similarity 
    # Second output = indicies of the images. 
    return distances, indices



# Save indices
def five_indices(distances, indices):
    idxs = [] # five closet images.
    for i in range(1,6):
        print(distances[0][i], indices[0][i])
        idxs.append(indices[0][i])
    return idxs


# Getting filepath for five closet images
def filepath_indices(filenames, idxs):
    filepaths = []
    for idx in idxs:
        filepaths.append(filenames[idx])
    return filepaths 

def creating_dataframe(filepaths, distances):
    # Create a dictionary keys filepath containing list filepaths, and key distance containg nearest neighbor distance
    five_closest_dictionary = {'Filepath': filepaths, 'Distance': distances[0][1:6]} # not including chosen image
    # Using pandas to create dataframe 
    five_closest_dataframe = pd.DataFrame(five_closest_dictionary)
    outpath = os.path.join("out", "nearest_neighbor_five_images.csv") # defining output path
    five_closest_dataframe.to_csv(outpath, index= False) # transforming my dataframe to a csv file, with no index.






def main_function():
    args = input_parse()
    model = pretrained_model()
    filenames = files_path(args)
    feature_list = feature_extraction(filenames, model)
    neighbors = nearest_neighbours(feature_list)
    distances, indices = calc_nearest_neighbour(args, neighbors, feature_list)
    idxs = five_indices(distances, indices)
    filepaths = filepath_indices(filenames, idxs)
    creating_dataframe(filepaths, distances)


if __name__ == "__main__":
    main_function()