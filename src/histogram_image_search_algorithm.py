# Importing Libraries 

# Data munging tools
import cv2
import pandas as pd
import zipfile
from tqdm import tqdm # Creates progress bars

# System tools
import os
import sys
sys.path.append("utils")
import argparse


# Functions created by me used in function: calculating_comparison
import helper_functions as hf

def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments // "--name" is what you feed it in the command line
    parser.add_argument("--choose_image", type=int, default = 28, help = "Number corresponding to images place in the index")
    parser.add_argument("--zip_name", type=str, help = "Path to the zip file")
    # parse the arguments from command line
    args = parser.parse_args()
    return args


def unzip(args):
    folder_path = os.path.join("data", "17_flowers") # Folder path
    if not os.path.exists(folder_path): # If the folder path does not exist, run the code below
        print("Unzipping file")
        path_to_zip = args.zip_name # Defining the path to the zip file
        zip_destination = os.path.join("data") # Defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination) # Where to place the unzipped files
    print("The files are unzipped")


# Get file paths for train and validation sets
def path_to_images(): # Joining the validation and train images together. There is no need for validation images in this script
    train_dir = os.path.join("data", "17_flowers", "train") # Train images
    validation_dir = os.path.join("data", "17_flowers", "validation") # Validation images
    
    filepaths = [] # Empty list to store all the filepaths
    print("Appending filepaths for train images")
    for root, dirs, files in os.walk(train_dir): # For entire file path for the train images append to empty list.
        for file in files:
            filepath = os.path.join(root, file)
            filepaths.append(filepath)

    print("Appending filepath for validation images")        
    for root, dirs, files in os.walk(validation_dir): # For entire file path for the validation images append to empty list
        for file in files:
            filepath = os.path.join(root, file)
            filepaths.append(filepath)
    
    filepaths = sorted(filepaths, key=lambda x: x.split("/")[-1]) # Sorting the list of filenames, based on the last part of the path
    print(filepaths)
    return filepaths
    

# Chosing image  
def choosing_image(args, filepaths):
    print("Selecting image")
    chosen_image = filepaths[args.choose_image] # Getting my chosen image for the list filepaths
    flower_image = cv2.imread(chosen_image) # using cv2s imread to read the image.
    print("Your chosen image path: ", chosen_image)
    print("Your chosen image: ", os.path.basename(chosen_image))
    
    return flower_image

# Creating histogram of chosen image
def flower_histogram(flower_image):
    print("Making histogram on chosen image")
    flower_hist = cv2.calcHist([flower_image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256]) # Calculating histogram of the image. 
    #  flower_image = image name, [0,1,2] = colour channels, none = no mask, [256,256,256] = colour channel size, [0,256...] = pixel amount
    return flower_hist


def flower_normalization(histogram):
    print("Normalizing chosen image")
    flower_hist_normalized = cv2.normalize(histogram, histogram, 0, 1.0, cv2.NORM_MINMAX) # normalizing the image so it easier to compute
    # the two first arguments, are the image we want to normalize, the two next arguments are the size we are changing the pixels to
    # the last argument is cv2's normalization calculations.
    
    return flower_hist_normalized


def calculating_comparison(flower_hist_normalized, filepaths):
    print("Processing all images")
    print("Creating Histogram for images")
    print("Normalizing images")
    print("Comparing images")
    print("Appending filename and score")
    img_score= [] # Creating empty list to store tuples of filename and comparison score for all images.
    print("Calculating comparison score...")
    for file_name in filepaths: # for each file in the list filepaths, run the code below

        if file_name.endswith(".jpg"): # only files the end with .jpg can move forward.
                
            flowers_img = cv2.imread(file_name) # reading the image using cv2's imread

            flowers_hist = hf.image_processing_histogram(flowers_img) # Creating histogram
            flowers_histnorm = hf.image_processing_normalization(flowers_hist) # Normalising
            comparing_score = hf.comparison(flower_hist_normalized, flowers_histnorm) # Comparing the image with the chosen image
            hf.filename_score(img_score, file_name, comparing_score) # Appending to the empty list
                
    return img_score


def five_lowest(img_score):  
    five_images = [] # empty list to store the five lowest scored images, and chosen image.
    print("Finding five lowest scores") 
    for i in range(6): # For loop that only runs six times.
        five_lowest_score = min(img_score, key=lambda x: x[1]) # The list contain all scores is a tuple of file_name and score. Here I am rearrange the list with the lowest scores first
        img_path, distance = five_lowest_score # Spliting into two variables
        filename = os.path.basename(img_path) # Selecting the last instance in the filepath. This is the image_001.jpg
        five_images.append((img_path, filename, distance)) # Appending img_path, filename and distance to my empty list.
        img_score.remove(five_lowest_score) # Removing the appended filename/score, so it does not get appended again.
    print("Done finding five lowest scores")
    return five_images


def save_dataframe(five_images):
    print("Saving five closest images")
    data = pd.DataFrame(five_images, columns=["Filepath", "Filename", "Distance"]) # Creating a dataframe 
    outpath = os.path.join("out", "histogram_five_images.csv") # defining output path
    data.to_csv(outpath, index= False) # transforming my dataframe to a csv file, with no index.


def main_function():
    args = input_parse() # Command line arguments
    unzip(args) # Unzipping function
    filepaths = path_to_images() # Creating list of filepaths to all images
    flower_image = choosing_image(args, filepaths) # Selecting an image 
    flower_hist = flower_histogram(flower_image) # Creating histogram of the image
    flower_hist_normalized = flower_normalization(flower_hist) # Normalising values
    img_score = calculating_comparison(flower_hist_normalized, filepaths) # Creating all comparisons
    five_images = five_lowest(img_score) # Finding the five closet images
    save_dataframe(five_images) # Saving the output


if __name__ == "__main__": # If called from command-line run main script.
    main_function()