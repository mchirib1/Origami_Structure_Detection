# Origami Structure Detection

Tools for dataset augmentation and analysis. These tools are used in conjuntion with the YOLOv5 object detection framework. (https://github.com/ultralytics/yolov5)

Datasets directory includes our source data for both triangle and breadboard as well as the accompanying annotatioins. 

## Augmentation Walkthrough

A walkthrough of our augmentation pipeline is available in the form of a Jupyter Notebook (Augmentation_notebook.ipynb). The src directory is the example source directory used in the walkthrough and it contains data from our triangle structure.

At the end of the walkthrough you will be able to copy the created data to google drive and download. Then it can be used for training a model and further detection. 

## Training and Detection Walkthrough

A Jupyter Notebook walkthrough using the YOLOv5 framework for training and detection is also available (YOLOv5s_notebook.ipynb). 

The data downloaded from the previous augmentation step can be used to create a training and validation set. Example testing data is available in the datasets directory. 

Pretrained weights for both the breadboard structure and the triangle structure are also included.

## Analysis

A MATLAB script we used for analysis of the results is also included in the matlab directory.
