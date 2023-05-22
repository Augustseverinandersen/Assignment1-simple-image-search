[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10267670&assignment_repo_type=AssignmentRepo)
# Building a simple image search algorithm

## 1.1 Assignment Description 
Written by Ross:
Define a particular image that you want to work with, for the image extract the colour histogram using OpenCv. Extract colour histograms for all the other images in the data. Compare the histogram of our chosen image to all the other histograms. For this, use the cv2.compareHist() function with the cv2.HISTCMP_CHISQR metric. Find the five images which are most similar to the target image. Save a CSV file to the folder called out, showing the five most similar images and the distance metrics:
|Filename|Distance|
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
1.3.1 Data
As mentioned previously, the data is created by Nilsback and Zisserman from the University of Oxford. The data contains images of 17 different categories of common flowers in the UK. According to the authors, the images "have large scale, pose and light variations.". This creates a challenge, as images of the same category, can look different based on how the picture was taken.
The data I am using is from Kaggle user Sani Kamal. The data is structured as so: ![image](https://github.com/Augustseverinandersen/Assignment1-simple-image-search/assets/112094086/e0b54d1e-5ebc-4e9e-a026-f1ef29ac0e19)


In total, each flower category has 80 flower images, which gives a total of over 1300 images of flowers. 








