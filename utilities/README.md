# Convolutional Neural Networks and Microplants
#### _utilities_ directory

This directory contains image preprocessing scripts (resize, reshape, downloading from a database), a simple program timer, and archived files.

---


## Layout

##### Folders:
- **archive/** - Contains an older version of the model and method of uploading images into the model that may still be explored in the future.
- **image_processing/** - Contains a tool to resize images, as well as scripts to download images from the Field Museum's digital collection of the *Lycopodieaceae* and *Selaginellaceae* families from an online database.  See the folder for full details.
- **Rehydrated/** - Contains all the rehydrated images of our plant specimens divided into subspecies and will hold our cropped and resized leaf photos 

##### Files:
- **timer.py**
  - Simple helper script to time and output execution of programs.
- **dataloader.py**
  - Set of functions to load/save files on the local file system.
- **load_zooniverse_results.py**
  - Parse a Herbarium Handwriting Transcription Zooniverse classifications export file 
    into a usable format for NN.
- **click_crop.py**
  - A way to select an image file and crop it automatically by selecting the upper left and lower right bounds of the desired portion
- **crop_resize.py**
  - Take the cropped images, find the largest width and height value between all cropped photos and resize each image to be the same size
- **cropped_image_data.csv**
  - Stores the height, width, and path of each cropped image