[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10267670&assignment_repo_type=AssignmentRepo)
# Building a simple image search algorithm
https://www.kaggle.com/datasets/sanikamal/17-category-flower-dataset
# Remember to give a path to the data



# Important to run script histo before feature
# Explain VGG16 (weights from imagenet)
# What are distance and indicies


## Contribution 
- The scripts in this repository were created with inspiration from in-class notebooks, and in collaboration with my fellow students. The data used in this assignment is accesed from [Kaggle](https://www.kaggle.com/datasets/sanikamal/17-category-flower-dataset), but created by [Maria-Elena Nilsback and Andrew Zisserman](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/), from the Visual Geometry Group at the University of Oxford.
# VGG16 weights trained from imagenet
- This assignment uses the pretrained convolutional neural network VGG16. VGG16 has 16 conolutional layers and is used for transfer learning.
### Data
- As mentioned previously, the data is created by Nilsback and Zisserman from the University of Oxford. The data contains image of 17 different categories of common flowers in the UK. According to the authors, the images *"have large scale, pose and light variations.".* This creates a challenge as multiple factors have an input on how similar the images look. 
- The zip file I am using is from Kaggle user Sani Kamal. The zip files contains a folder called *17_flowers*, which has two folders *train* and *validation*. Train and validation contain 17 folders each, of the 17 categories of flowers. In total each flower category has 80 flower images. In total there is over 1300 images of flowers.
|Bluebell|Buttercup|Colts Foot|Cowslip|Crocus|Daffodil|Daisy|Dandelion|Fritillary|Iris|Lily of the Valley|Pansy|Snowdrop|Sunflower|Tigerlily|Tulip|Windflower|
  
## Packages
- Cv2
  - cv2 is being used to create histograms, normalize, and compare.
- Pandas
  - Pandas is being used to create a dataframe for the output 
- Zipfile
  - Zipfile is being used to unpack the zipfile.
- Tqdm
  - Is being used to create a progress bar for a for loop.
- Os
  - Is used to navigate filepaths.
- Sys
  - Is used to navigate the directory.
- Argparse
  - Is used to create command-line arguments.
- Numpy
  - From numpy the *norm* function is being imported. Norm is used to normalize the feature vector in the *helper_function.py* script
- Tensorflow
  - From Tensorflow the following is being imported *Hub, load_img, img_to_array, VGG16,* and *preprocess_input.*
    - Hub is being used to get VGG16
    - Load_img is used to load an image 
    - Img_to_array is being used to make the image to an array
    - VGG16 is the pretrained model
    - preprocess_input is used to make the images compatiable with VGG166
- Scikit Learn
  - From scikit learn nearestNeighbors is being imported, to use the feature extractions and find the most similar images.

## Repository Contents
- The repository contains the following folders and files 
  - assignment_1 
    - The virtual envrionment 
  - data
    - An empty folder where you will place the zip file 
  - out 
    - Here the csv files from the two scripts will be saved. The csv file for the script *histogram_image_search_algorithm.py* contains *filename, filepath,* and *distance*, for the six closet images. This csv file also includes the chosen image, as the first row in the csv file. The csv file for the script *feature_extraction_image_search_algorithm.py*, contains *filepath* and *distance*, for the five closets images, but does not include the chosen image.
  - src 
    - The folder that contains the two scripts. 
      - *histogram_image_search_algorithm.py* uses histogram comparison to find the most similar images.
      - *feature_extraction_image_search_algorithm.py* uses VGG16 and feature extraction to find the most similar images.
  - utils
    - This folder contains a script called *helper_functions.py*, which contains functions used in both scripts. The first four functions are used in the histogram script, to create a histogram, normalise it, compare, and append. The last function is used in the feature extraction script.
  - README.md
    - The README file
  - requirements.txt 
    - This text file has the packages that get installed when runing the setup.sh
  - setup.sh
    - This is the setup file, which creates a virtual environment, installs the requirements, and upgrades pip.

## Machine Specifications and My Usage
- This script was created on Ucloud. It took 15 minutes to unzip the zip file and run both scripts, on a 16-CPU machine.

## Assignment Description
Assignment description_

- Define a particular image that you want to work with
- For that image
  - Extract the colour histogram using ```OpenCV```
- Extract colour histograms for all of the **other* images in the data
- Compare the histogram of our chosen image to all of the other histograms 
  - For this, use the ```cv2.compareHist()``` function with the ```cv2.HISTCMP_CHISQR``` metric
- Find the five images which are most simlar to the target image
  - Save a CSV file to the folder called ```out```, showing the five most similar images and the distance metric:

|Filename|Distance]
|---|---|
|target|0.0|
|filename1|---|
|filename2|---|

-  Create a script for image searching using a pretrained CNN.


## Methods / What the code does
Script *histogram_image_search_algorithm.py*: 
- This scripts starts by unzipping the zipfile and creates a list, which stores the filepath to each image. Next you can use argparse to select a filepath from the sorted list. The script than uses cv2s function ```calcHist``` to create a histogram of your chosen image, before normalizing it. To normalize the script used cv2s ```normalize``` function with the argument *cv2.NORM_MINMAX.* This argument normalizes the pixel values between to given numbers. I have chosen zero to one. After normalizing the script than compares the chosen image with all other images to get a distances score. The functions are accessed from the *utils* folder, in the script *helper_functions.py.* One of the functions is cv2s ```compareHist```. Which takes two images and compares them using *cv2.HISTCMP_CHISQR.* This argument calculates the Chi-Square distance between the two histograms, the lower the score the smiliar the images are. Lastly, the script creates a pandas dataframe of the six closet images (including the chosen image), and saved it as a csv file in the folder *out*.

Script *feature_extraction_image_search_algorithm.py*:
- This script starts by loading the pretrained model VGG16 without classification layers, before creating a sorted list of all image filepaths. Next the function for feature extraction is run, which returns a normalized feature vector for each image. This function utilies a function in the *helper_functions.py* script, which is used with VGG16 to get the features of the images. Next, scikit learns algorithm, ```NearestNeighbors```, is used to find the ten closet images, from the features created before. The algorithm finds the most similar images based on cosine distance. The script than calls the algorithm, based on the chosen image to find the *distance* and *indices* of the other images. The distance is the cosine similarity. The smaller the distance the similar the image. The indicies is which image. Lastly, the script finds the five closet images, and the filepath, before saving the output as a csv file in the folder *out*. The chosen image is also printed to the command-line, so you can see what image you chose and how it looks.

## Discussion 
- Both methods of finding similar images are useful, but using feature extraction and a pretrained model is clearly better. When I ran the histogram script I chose image_003.jpg. The closet image had a score of 3155, the closer to zero the more similar, and the other four images had a score of over 3200. When tested on other images, it performs better with closer scores for all five images, but none as close as the feature extraction script. The reason that the histogram approach for finding similar images, is not as good, is because all the background colour, which is not the flower, is included in the histogram. This can cause two images of the same flower, but with different colour backgrounds, to have far apart scores. Moreover, a histogram approach only takes colour as an input, and not other features which could be present for a specific flower or image. When I ran the feature extraction script on the same image, all the scores were close together. The reason that feature extraction is a much more powerfull way of finding similiar images is because feature extraction, takes more representations of the images, and is not as effected in changes to light or colour. Moreover, feature extraction in this case is using a pretrained model, which has been trained on a range of different images, to get specific weights for predicting a correct image.


## Usage
To run the scripts in this repository follow these steps:
__OBS!__ Important to start with *histogram_image_search_algorithm.py*, as this is the script that unzips the zip file.

- *histogram_image_search_algorithm.py* script:
  - Get the zip file from [Kaggle](https://www.kaggle.com/datasets/sanikamal/17-category-flower-dataset), and place it in the data folder.
  - Run ```bash setup.sh``` in the command-line, which will create a virtual environment, and install the requirements 
  - Run ```source ./assignment_1/bin/activate``` in the command-line, to activate the virtual environment.
  - In the command-line run ```python3 src/histogram_image_search_algorithm.py --choose_image 2 --zip_name data/archive.zip```
    - ```--choose_image``` = Your chosen image. The number is the placement in the filepath list. A default number is given.
    - ```--zip_name``` = Your path to the zip file.

- *feature_extraction_image_search_algorithm.py* script:
  - Run ```python3 src/feature_extraction_image_search_algorithm.py --choose_image 2```
    - ```--choos_image``` = Your chosen image.
