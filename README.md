# 3D Object Localisation Project
Adapted implementation of the paper "CAD-based recognition of 3D objects in monocular images" for the M2 Computer Vision course at Sapienza University.
Link: https://www.researchgate.net/publication/221072299_CAD-based_recognition_of_3D_objects_in_monocular_images

A project done by Jean-Louis MATERNA and Tristan DESJARDINS.

## General idea  

The idea is to find the coordinates of an object based on an image (test image). To do so, we generate images (camera views) of the object under different angles. We then, try to find which one of these camera views is the most similar to our test image (using a similarity score). 

However, instead of comparing our test image to all the camera views, we create a sort of "tree" (pyramid) where at each iteration, a bunch of camera views are excluded. Therefore, it works much faster than a naive comparison with all the camera views. Once we found the best matching one, we assume the coordinates of our test image are those of the most similar camera view (which we know since we've generated it).  

## Previews

Here are a few images from our code. 

### Generation of the camera views

We can see the generated camera views (images of our object under different angles). 
<img src="https://user-images.githubusercontent.com/62900180/154372769-5c2147b2-67d4-4b44-8040-88baf28e94b2.png" alt="drawing" height="300"/>

### Generation of the Pyramid

Here we can see the different levels of the pyramid. Each white square represent a camera view. We start with level 4 all the way to level 1, excluding camera views at each new level until we find the one that matches the most. 
<img src= "https://user-images.githubusercontent.com/62900180/154373124-ed86bbae-f601-4a06-99dd-9b12d97cad57.png" alt="drawing" height="250"/>

### Testing our Pyramid (model)

Here, the whiter a square (camera view) is, the more similarity there is with the test image. For instance, for the first image, we used a test image where the object position is at (0,0). Therefore, we expect more similarity with camera views at (0,0). We can see that the closer we get to (0,0) the whiter the squares get (which means more similarity), proving our algorithm works. 

<img src= "https://user-images.githubusercontent.com/62900180/154373440-254234a1-6aed-4fe7-a0c8-30771b5d7ae8.png" alt="drawing" height="250"/>

## Code
There are 5 .py files in the project :
- main.py : all the main functions used
- training_data_gen.py : camera views generation
- offline_params : definition of all the parameters used in the training phase (path etc)
- wavefront.py : importing the model from Blender


