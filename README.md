[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10267670&assignment_repo_type=AssignmentRepo)
# Building a simple image search algorithm

## 1.1 Assignment Description 
Written by Ross:
Define a particular image that you want to work with, for the image extract the colour histogram using OpenCv. Extract colour histograms for all the other images in the data. Compare the histogram of our chosen image to all the other histograms. For this, use the cv2.compareHist() function with the cv2.HISTCMP_CHISQR metric. Find the five images which are most similar to the target image. Save a CSV file to the folder called out, showing the five most similar images and the distance metrics:

|Filename|Distance]
|Target|0.0|
|Filename1|--|
|Filename2|--|

Create a script for image searching using a pre-trained CNN.
## 1.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. Python version 1.73.1. It took 15 minutes to unzip the zip file and run both scripts, on a 16-CPU machine. 
## 1.2.1 Prerequisites 
To run the scripts, make sure to have Bash and Python 3 installed on your device. The scripts have only been tested on Ucloud. 
## 1.3 Contribution
The scripts in this repository were created in collaboration with fellow students. The data used in this assignment is accessed from [Kaggle](https://www.kaggle.com/datasets/sanikamal/17-category-flower-dataset), but created by [Maria-Elena Nilsback and Andrew Zisserman](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/), from the Visual Geometry Group at the University of Oxford.
This assignment uses the pre-trained convolutional neural network VGG16. 
### 1.3.1 Data
As mentioned previously, the data is created by Nilsback and Zisserman from the University of Oxford. The data contains images of 17 different categories of common flowers in the UK. According to the authors, the images "have large scale, pose and light variations.". This creates a challenge, as images of the same category, can look different based on how the picture was taken.
The data I am using is from Kaggle user Sani Kamal. The data is structured as so: ![image](https://github.com/Augustseverinandersen/Assignment1-simple-image-search/assets/112094086/e0b54d1e-5ebc-4e9e-a026-f1ef29ac0e19)

In total, each flower category has 80 flower images, which gives a total of over 1300 images of flowers. 
## 1.4 Packages
These are the packages that I used to make the scripts:
•	Cv2 (version 4.7.0.72) is being used to create *histograms*, *normalize*, and *compare* the images.
•	Pandas (version 2.0.1) is being used to create a data frame for the output.
•	Zipfile is being used to unpack the zip file.
•	Tqdm (version 4.65.0) is used to create a progress bar for a for loop.
•	Os is used to navigate file paths, on different operating systems.
•	Sys is used to navigate the directory.
•	Argparse is used to create command-line arguments.
•	NumPy (version 1.23.5) from NumPy the *norm* function is being imported. Norm is used to normalize the feature vector in the ```helper_function.py script```.
•	TensorFlow (version 2.12.0 & 0.13.0) from TensorFlow the following is being imported: _Hub_ is being used to get _VGG16_. _Load_img_ is used to load an image. _Img_to_array_ is being used to make the image into an array. _VGG16_ is the pre-trained model. _preprocess_input_ is used to make the images compatible with VGG16.
•	Scikit-learn (version 1.2.2) from scikit-learn _nearestNeighbors_ is being imported, to use _feature extractions_ and find the most similar images.
## 1.5 Repository Contents
The repository contains the following folders and files.
•	***data*** an empty folder where the zip file will be placed.
•	***out*** the folder where the CSV files from the two scripts will be saved. 
  o	The CSV file for the script ```histogram_image_search_algorithm.py``` contains _filename_, _file path_, and _distance_, for the six closet images. This CSV file also includes the chosen image, as the first row in the CSV file. 
  o	The CSV file for the script ```feature_extraction_image_search_algorithm.py```, contains _file path_ and _distance_, for the five closets images, but does not include the chosen image.
•	***Src*** the folder that contains the two scripts:
  o	```histogram_image_search_algorithm.py``` uses _histogram comparison_ to find the most similar images.
  o	```feature_extraction_image_search_algorithm.py``` uses VGG16, _nearestNeighbors_, and _feature extraction_ to find the most similar images.
•	***utils*** this folder contains a script called ```helper_functions.py```, which contains functions used in both scripts. The first four functions are used in the histogram script, to create a _histogram, normalize it, compare_, and _append_. The last function is used in the feature extraction script.
•	***README.md*** the README file.
•	***requirements.txt*** this text file contains the packages that will be installed when running the setup.sh.
•	***setup.sh*** this is the setup file, which creates a virtual environment, installs the requirements, and upgrades pip.
## 1.6 Methods 
### 1.6.1 ```Script histogram_image_search_algorithm.py```:
•	This script starts by unzipping the zip file and creates a list, of each image path. 
•	With argparse, you can select a file path from the sorted list, by writing a list index. 
•	The script then uses cv2s function, _calcHist_, to create a histogram of your chosen image, before _normalizing_ it. 
•	To normalize, the script uses cv2s _normalize_ function with the argument _cv2.NORM_MINMAX_. This argument normalizes the pixel values between two given numbers. I have chosen zero to one. By normalizing, the images become easier to compare.
•	After normalizing, the script then creates a _histogram_, normalizes all other images, and compares the images with the chosen image. This is done by using functions located in the script ```helper_functions.py```, in the folder utils.
•	The function for comparing uses cv2s _compareHist_. Which takes two images and compares them using _cv2.HISTCMP_CHISQR_. This argument calculates the Chi-Square distance between the two histograms. The lower the score the more similar the images.
•	Lastly, the script creates a Pandas data frame of the six closet images (including the chosen image) and saves it as a CSV file in the folder out.
### 1.6.2 Script ```feature_extraction_image_search_algorithm.py```:
•	This script starts by loading the pre-trained model _VGG16_ without classification layers.
•	The script then creates a sorted list of all paths to the images.
•	Next, the function for feature extraction is run, which returns a normalized feature vector for each image. This function utilizes a function in the ```helper_functions.py``` script, which is used with _VGG16_ to get the features of the images.
•	Next, Scikit-Learns algorithm, _nearestNeighbors_, is used to find the ten most similar images based on cosine distance. 
•	The script then uses the _nearestNeighbor_ algorithm, based on the chosen image to find the distance and indices of the other images. The distance is the cosine similarity. The smaller the distance the more similar the image. The indices are which image. 
•	Lastly, the script finds the five closet images, and the file path, before saving the output as a CSV file in the folder _out_. The chosen image path is also printed to the command line, so you can see find your chosen image.
## 1.7 Discussion
Both methods of finding similar images are useful, but using feature extraction and a pre-trained model is better. When I ran the histogram script, I chose _image_003.jpg_ (list index 2). The closest image had a score of 3155 (the closer to zero the more similar), and the other four images had a score of over 3200. When tested on other images, it performed better with scores closer to zero, but none as close as the feature extraction script. The five closest images in the feature extraction script were all under 1.

The reason that the histogram approach for finding similar images is not as effective, is because all the background colour is included in the histogram. This can cause two images of the same flower, but with different colour backgrounds, to have far apart scores. Moreover, a histogram approach only takes colour as an input, and not other features which could be present for a specific flower or image. 

When I ran the feature extraction script on the same image, all the scores were close together. The reason that feature extraction is a much more powerful way of finding similar images is that feature extraction takes more representations of the images, and is not as affected by changes to light or colour. Moreover, feature extraction, in this case, is using a pre-trained model, which has been trained on a diverse range of images, to get specific weights for predicting a correct image.
## 1.8 Usage
To run the scripts in this repository, follow these steps: 
OBS! Important to start with ```histogram_image_search_algorithm.py```, as this is the script that unzips the zip file.
•	```histogram_image_search_algorithm.py``` script:
  1.	Clone the repository.
  2.	Get the zip file from [Kaggle](https://www.kaggle.com/datasets/sanikamal/17-category-flower-dataset), and place it in the data folder.
  3.	Run ```bash setup.sh``` in the command line. This will create a virtual environment and install the packages in the requirements.txt file.
  4.	Run ```source ./assignment_1/bin/activate in the command-line```, to activate the virtual environment. 
  5.	In the command-line run ```python3 src/histogram_image_search_algorithm.py --choose_image 2 --zip_name data/archive.zip``` to run the histogram script.
    • The argparse ```--choose_image``` takes an integer number. This number corresponds to an index in the list. The default is 28, which corresponds to image_029.jpg. 
    • The argparse ```--zip_name``` takes a string value. Here you should write the path to your zip file. Your zip file must be placed in the data folder. Thereby, you will write data/ZIP_FILE_NAME.zip.
  6.	Run ```python3 src/feature_extraction_image_search_algorithm.py --choose_image 2``` to run the feature extraction script.
    •	The argparse ```--choose_image``` takes an integer number. This number corresponds to an index in the list. The default is 28, which corresponds to image_029.jpg. 








