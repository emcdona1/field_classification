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
- **prepare_csv_data.py**
  - Creates cleaned and voted csvs from given csv (made for Zooniverse)
- **vote_csv_data.py**
  - Helper functions for prepare_csv_data.py
- **sort_images_with_voted.py**
  - Sorts given images into categories based on voted csv