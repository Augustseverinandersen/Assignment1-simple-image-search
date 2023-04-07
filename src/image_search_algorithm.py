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
    parser.add_argument("--image_name", type=str)
    parser.add_argument("--directory", type=str )
    # parse the arguments from command line
    args = parser.parse_args()
    return args




# See if you can add an if statement here that sees if it is already unzipped
def unzip():
    folder_path = os.path.join("data", "flowers")
    if not os.path.exists(folder_path):
        print("Unzipping file")
        path_to_zip = os.path.join("data", "flowers.zip") # Defining the path to the zip file
        zip_destination = os.path.join("data", "flowers") # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination)
    print("The files are unzipped")



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
def choosing_image(args):
    print("Selecting image")
    image = os.path.join("data", "flowers", "flowers", args.image_name) # Defining the path to my chosen image
    chosen_image = os.path.basename(image) # Using os.path.basename to get the last component of the path. The last component is the img name 
    # I am doing this, to exclude this image in my function when comparing histograms.
    flower_image = cv2.imread(image) # using cv2s imread to read the image.
    print("My chosen image: ", chosen_image) # Checking if I have got the image name as a string 
    
    return flower_image, chosen_image


def flower_histogram(image):
    print("Making histogram on chosen image")
    flower_hist = cv2.calcHist([image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256]) # Calculating histogram of the image. 
    #  flower_image = image name, [0,1,2] = colour channels, none = no mask, [256,256,256] = colour channel size, [0,256...] = pixel amount
    return flower_hist

def flower_normalization(histogram):
    print("Normalizing chosen image")
    flower_hist_normalized = cv2.normalize(histogram, histogram, 0, 1.0, cv2.NORM_MINMAX) # normalizing the image so it easier to compute
    # the two first arguments, are the image we want to normalize, the two next arguments are the size we are changing the pixels to
    # the last argument is cv2's normalization calculations.
    
    return flower_hist_normalized



def image_paths(args, chosen_image, flower_hist_normalized):
    print("Processing all images")
    print("Creating Histogram for images")
    print("Normalizing images")
    print("Comparing images")
    print("Appending filename and score")
    directory = args.directory
    img_score= [] # Creating empty to store tuples of filename and comparison score for all images.
    print("Calculating comparison score...") # Checking if the function is running.
    for file_name in os.listdir(directory): # using os.listdir, to create a path to each image, when I define the directory.
        
        if file_name != chosen_image: # excluding my chosen image (the variable I created earlier).

            if file_name.endswith(".jpg"): # only files the end with .jpg can move forward.
                   
                file_path = os.path.join(directory, file_name) # joing the path for each file to the directory.
                flowers_img = cv2.imread(file_path) # reading the image using cv2's imread
    
                flowers_hist = image_processing_histogram(flowers_img)
                flowers_histnorm = image_processing_normalization(flowers_hist)
                comparing_score = comparison(flower_hist_normalized, flowers_histnorm)
                filename_score(img_score, file_name, comparing_score)
                
    return img_score

def five_lowest(img_score):  
    five_images = [] # empty list to store the five lowest scored images. 
    print("Finding five lowest scores") # Printing a message to let me know where in the function I am.
    for i in range(5): # For loop that only runs five times.
        five_lowest_score = min(img_score, key=lambda x: x[1])
        # Using min to find the lowest score. lambda is a function in python. I am using it here to find the min value of the second part,
         # Of each tuple in my list.
        five_images.append(five_lowest_score) # Appending the lowest score to my empty list.
        img_score.remove(five_lowest_score) # Removing the appended filename/score, so it does not get appended again.
    print("Done finding five lowest scores") # Printing an update message.
    return five_images

def save_dataframe(five_images):
    print("Saving five lowest images")
    data = pd.DataFrame(five_images, columns=['Filename', 'Score']) # Creating a dataframe with the two columns Filename and Score
    outpath = os.path.join("out", "five_images.csv") # defining output path
    data.to_csv(outpath, index= False) # transforming my dataframe to a csv file, with no index.


def main_function():
    args = input_parse()
    unzip()
    flower_image, chosen_image = choosing_image(args)
    flower_hist = flower_histogram(flower_image)
    flower_hist_normalized = flower_normalization(flower_hist)
    img_score = image_paths(args, chosen_image, flower_hist_normalized)
    five_images = five_lowest(img_score)
    save_dataframe(five_images)



if __name__ == "__main__":
    main_function()























