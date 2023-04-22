# Importing Libraries 

# path tools
import os
import numpy as np
import cv2
import pandas as pd
import sys
sys.path.append("utils")
from imutils import jimshow
from imutils import jimshow_channel
import matplotlib.pyplot as plt
import zipfile
import argparse



def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments // "--name" is what you feed it in the command line
    parser.add_argument("--image_index", type=int, default = 28)
    parser.add_argument("--directory", type=str )
    parser.add_argument("--zip_name", type=str)
    # parse the arguments from command line
    args = parser.parse_args()
    return args





def unzip(args):
    folder_path = os.path.join("data", "17_flowers")
    if not os.path.exists(folder_path):
        print("Unzipping file")
        path_to_zip = args.zip_name # Defining the path to the zip file
        zip_destination = os.path.join("data") # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination)
    print("The files are unzipped")
        # Get file paths for train and validation sets

    train_dir = os.path.join("data", "17_flowers", "train")
    validation_dir = os.path.join("data", "17_flowers", "validation")
    
    filepaths = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            filepath = os.path.join(root, file)
            filepaths.append(filepath)
            
    for root, dirs, files in os.walk(validation_dir):
        for file in files:
            filepath = os.path.join(root, file)
            filepaths.append(filepath)
    return filepaths
    



 # Functions to be used later not included in main function
def image_processing_histogram(images):
    # creating histogram for the img
    flowers_hist = cv2.calcHist([images], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    return flowers_hist

def image_processing_normalization(images):
    # Normalising the img
    flowers_histnorm = cv2.normalize(images, images, 0, 1.0, cv2.NORM_MINMAX)
    return flowers_histnorm

def comparison(image, images):
    # Comparing the img score with my chosen image
    comparing_score = round(cv2.compareHist(image, images, cv2.HISTCMP_CHISQR), 2)
    return comparing_score


def filename_score(empty_list, file_name, score):
    # Saving the filename and the score, in my empty list as a tuple.
    empty_list.append((file_name, score))





# Chosing image function 
def choosing_image(args, filepaths):
    print("Selecting image")
    chosen_image = filepaths[args.image_index] 
    # I am doing this, to exclude this image in my function when comparing histograms.
    flower_image = cv2.imread(chosen_image) # using cv2s imread to read the image.
    print("My chosen image path: ", chosen_image) # Checking if I have got the image name as a string 
    print("My chosen image: ", os.path.basename(chosen_image))
    
    return flower_image, chosen_image


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



def image_paths(args, flower_hist_normalized, filepaths):
    print("Processing all images")
    print("Creating Histogram for images")
    print("Normalizing images")
    print("Comparing images")
    print("Appending filename and score")
    img_score= [] # Creating empty to store tuples of filename and comparison score for all images.
    print("Calculating comparison score...") # Checking if the function is running.
    for file_name in filepaths: # using os.listdir, to create a path to each image, when I define the directory.
        
        #if file_name != chosen_image: # excluding my chosen image (the variable I created earlier).

        if file_name.endswith(".jpg"): # only files the end with .jpg can move forward.
                
            flowers_img = cv2.imread(file_name) # reading the image using cv2's imread

            flowers_hist = image_processing_histogram(flowers_img)
            flowers_histnorm = image_processing_normalization(flowers_hist)
            comparing_score = comparison(flower_hist_normalized, flowers_histnorm)
            filename_score(img_score, file_name, comparing_score)
                
    return img_score

def five_lowest(img_score):  
    five_images = [] # empty list to store the five lowest scored images. 
    print("Finding five lowest scores") # Printing a message to let me know where in the function I am.
    for i in range(6): # For loop that only runs five times.
        five_lowest_score = min(img_score, key=lambda x: x[1])
        img_path, distance = five_lowest_score
        filename = os.path.basename(img_path)
        # Using min to find the lowest score. lambda is a function in python. I am using it here to find the min value of the second part,
         # Of each tuple in my list.
        five_images.append((img_path, filename, distance)) # Appending the lowest score to my empty list.
        img_score.remove(five_lowest_score) # Removing the appended filename/score, so it does not get appended again.
    print("Done finding five lowest scores") # Printing an update message.
    return five_images

def save_dataframe(five_images):
    print("Saving five closest images")
    data = pd.DataFrame(five_images, columns=["Filepath", "Filename", "Distance"]) # Creating a dataframe 
    outpath = os.path.join("out", "histogram_five_images.csv") # defining output path
    data.to_csv(outpath, index= False) # transforming my dataframe to a csv file, with no index.


def main_function():
    args = input_parse()
    filepaths = unzip(args)
    flower_image, chosen_image = choosing_image(args, filepaths)
    flower_hist = flower_histogram(flower_image)
    flower_hist_normalized = flower_normalization(flower_hist)
    img_score = image_paths(args, flower_hist_normalized, filepaths)
    five_images = five_lowest(img_score)
    save_dataframe(five_images)



if __name__ == "__main__":
    main_function()