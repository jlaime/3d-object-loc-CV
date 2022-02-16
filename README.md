# 3D Object Localisation Project
Adapted implementation of the paper "CAD-based recognition of 3D objects in monocular images" for the M2 Computer Vision course at Sapienza University.
Link: https://www.researchgate.net/publication/221072299_CAD-based_recognition_of_3D_objects_in_monocular_images

## General idea  

The idea is to find the coordinates of an object based on an image (test image). To do so, we generate images (camera views) of the object under different angles. We then, try to find which one of these camera views is the most similar to our test image (using a similarity score). 

However, instead of comparing our test image to all the camera views, we create a sort of "tree" (pyramid) where at each iteration, a bunch of camera views are excluded. Therefore, it works much faster than a naive comparison with all the camera views. Once we found the best matching one, we assume the coordinates of our test image are those of the most similar camera views (which we know since we've generated it).  

## Previews

Here are a few images from our code. 

### Generation of the camera views

### Generation of the Pyramid

### Testing our Pyramid (model)

## Code
There are 5 .py files in the project :
- main.py : all the main functions used
- training_data_gen.py : camera views generation
- offline_params : definition of all the parameters used in the training phase (path etc)
- wavefront.py : importing the model from Blender

A project done by Jean-Louis MATERNA and Tristan DESJARDINS.
