# Scripts for the _Rhododendron_ leaf classification project

### Using `click_crop.py`

This file allows the user to crop images by clicking on the upper left and lower right bounds of the desire segment.

When the program is run it asks the user to select a photo they wish to crop. It takes both .jpg and .png files. 
Once the photo is selected the user will click first in the upper lefthand corner of their desire crop area then 
in the lower righthand corner of the crop area. The program will take those coordinates and make a rectangular 
area out of those values to then crop the image from. After all desired sections have been selected, close the 
picture window and the photos will be saved to the folder the original photo came from as well as save the image 
data to crop_image_data.csv for padding later.

The program takes no parameters and can be run by simply executing `python click_crop.py`

---

### Using `auto_crop.py`

This file takes in an image file then automatically segments and crops it.

Note: This program is specifically designed for the imaging style of our Field Museum images, it can be used 
for other imaging but adjustments will need to be made.

When the program is run it asks the user to select a photo they wish to crop. It takes both .jpg and .png files. 
Once the photo is selected the program will automatically threshold the image and detect the edges of the leaves. 
It will then display the image with each segment highlighted with a red box. After closing the image window the 
program will begin to automatically crop, save, and write the image data to the auto_crop_image_data.csv file.

The program takes no parameters and can be run by simply executing `python auto_crop.py`

---

### Using `crop_resize.py`

This files uses the image data stored in a CSV file and resizes the images to be as large as the largest on in the 
file by padding it with whitespace.

Before running the program you need to have a CSV file filled with all the image data for the images you wish 
to resize. The image data should be stored on separate lines representing length, width, and file path to the 
image (in that order). When the program is run it will scan through the desired CSV file and find the largest 
values for both height and width of the image and begin to resize all images stored in the file to that size 
using white padding. The images are padded using their filepath saved in the CSV file.

---

### Using `rhododendron_image_augmentor.py`

This file loads in two folders of images, one training set and one testing set, and augments them using the 
chosen augmentation functions. The user will specify the file path for both training and testing folders as 
well as the batch size for both. Different augmentations are written in the code that can be turned on and off 
to try new combinations of augmentation. This script can be run in the terminal as follows:

`python rhododendron_image_augmentor.py <training_folder_path> <training_batch_size>, <testing_folder_path>, <testing_batch_size>`

Example: `python rhododendron_image_augmentor.py training_images 100 testing_images 20`

---

### Using `only_abaxial_images.py`

This file allows you to search through an image folder and extract images with a certain word in the title. 
In this case we use it to extract only the abaxial leaf images. The script takes one parameter which is just the 
file path of the image folder.  

Example: `python only_abaxial_images.py \5_leaf_categories\category1\hippophaeoides`
