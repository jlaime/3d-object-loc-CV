# 3D Object Localisation Project
Adapted implementation of the paper "CAD-based recognition of 3D objects in monocular images" for the M2 Computer Vision course at Sapienza University.
Link: https://www.researchgate.net/publication/221072299_CAD-based_recognition_of_3D_objects_in_monocular_images

A project done by Jean-Louis MATERNA and Tristan DESJARDINS.

## General idea  

The idea is to locate accurately a real object inside a picture, with the help of the 3D object's model. To do so, we generate numerous camera views of the 3D model under different angles. Then, we try to find which one of these camera views is the most similar to our test image, taking the position variations and lens deformations into account. 

This approach, while exhaustive, is naive and does not suit to realtime applications. 
Therefore, instead of comparing our input picture to all the camera views, we create a sort of "tree" (pyramid) where at each iteration, most of camera views are excluded. Once the best camera coordinates are found, we assume the coordinates of our test image are those of the most similar camera view (which we know since we've generated it).  

## Previews

Here are a few images from our code. 

### Generation of the camera views

We can see the generated camera views (images of our object under different angles). 
<img src="https://user-images.githubusercontent.com/62900180/154372769-5c2147b2-67d4-4b44-8040-88baf28e94b2.png" alt="drawing" height="300"/>

### Generation of the Pyramid

Here, we can see the different levels of the pyramid. Each white square represents a camera view. We start with level 4 all the way to level 1, excluding camera views at each new level until we find the one that matches the most. 
<img src= "https://user-images.githubusercontent.com/62900180/154373124-ed86bbae-f601-4a06-99dd-9b12d97cad57.png" alt="drawing" height="250"/>

### Testing our Pyramid (model)

Here, the whiter a square (camera view) is, the more similarity there is with the test image. For instance, for the first image, we used a test image where the object position is at (0,0). Therefore, we expect more similarity with camera views at (0,0). We can see that the closer we get to (0,0) the whiter the squares get (which means more similarity), proving our algorithm works. 

<img src= "https://user-images.githubusercontent.com/62900180/154373440-254234a1-6aed-4fe7-a0c8-30771b5d7ae8.png" alt="drawing" height="250"/>

## Code
There are 5 .py files in the project :
- main.py : all the main functions used
- training_data_gen.py : camera views generation
- offline_params.py : definition of all the parameters used in the training phase (path etc)
- wavefront.py : importing the model from Blender


