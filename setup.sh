#!/usr/bin/env bash

# Create virtual enviroment 
python3 -m venv assignment_2

# Activate virtual enviroment 
source ./assignment_2/bin/activate 

# Installing requirements 
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Run the code 
python3 code/image_search_algorithm.py --image_name image_0201.jpg --directory data/flowers/flowers

# Deactivate the virtual environment.
deactivate