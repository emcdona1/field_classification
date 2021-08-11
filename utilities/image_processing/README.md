# Convolutional Neural Networks and Microplants
#### _utilities\image_processing_ directory

The code here creates and tests a CNN model using Tensorflow and Keras that takes images of two morphologically similar plant families (*Lycopodieaceae* and *Selaginellaceae*) and trains the model to identify which is which. 

# Downloading and processing images, and training the model

## Using image_download.py
**Note**: The file `Download_Image_Files_from_Pteridoportal.ipynb` is a standalone version 
of this program which is designed to be run in the browser by non-developers.  :)

The goal of this file is to download images from the Pteridophyte Portal. Before running the file, please follow the steps below:
1. Go to [The Pteridophyte Collections Consortium](http://www.pteridoportal.org/portal/)
2. Click Search > Collections > Deselect All > Choose your source (we chose 'Field Museum of Natural History Pteridophyte Collection')
3. Hit Search in the upper right
4. Fill in your search parameters and hit 'List Display'
5. In the top right, click the little download button (it looks like a down arrow into a open box)
6. Choose the following parameters:

* Structure: Darwin Core  
* Data Extensions: Keep both boxes selected   
* File Format: Comma Delimited (CSV)   
* Character Set: UTF-8 (Unicode)   
* Compression: Check this box   
   
7. Hit 'Download Data'

Once you have your downloaded zip file, you will want two CSV's in particular: the images and occurrences
Place these files in the folder that your code is in and create a new folder in this space where you want the images to download.

Run this code by using the following command in terminal/command prompt

`python image_download.py -f [image_csv_name].csv -o [occurrences_csv_name].csv -l [folder_name]`

For example:

`python image_download.py -f images.csv -o occurrences.csv -l specimen_images`

`python image_download.py -f lyco_images.csv -o lyco_occurrences.csv -l lyco_images`

or if the CSVs are in a folder inside the workspace:

`python image_download.py -f lyco_csvs/lyco_images.csv -o lyco_csvs/lyco_occurrences.csv -l lyco_images`

Because of efficiency purposes, the program first looks in the 'identifier' column of the first csv for the image. If it's not found, it will then look in the 'goodQualityAccessURI' column. If neither produce a useable image, the program will output a CSV with the missing images in the folder you input that lists the barcodes and core id numbers.

## Using image_resize.py

The purpose of this file is to take the raw downloaded files and convert them into squares of the same size using an image processing package called OpenCV for Python. To install the package, check out [this website](https://pypi.org/project/opencv-python/) 

The program takes in the folder with all the original images, the destination for the resultant images, and the desired image size (just as a number because it's resizing to a square). The default size is 256 x 256 pixels, but it can be changed.

To use the file, follow this format in the command line terminal:

`python image_resize.py -sf orig_image_folder_path -df dest_folder_path -s image_size`

For example:
`python image_resize.py -sf orig_images -df smaller_images -s 256`

## Using click_crop.py

This file allows the user to crop images by clicking on the upper left and lower right bounds of the desire segment

When the program is run it asks the user to select a photo they wish to crop. It takes both .jpg and .png files. Once the photo is selected the user will click first in the upper lefthand corner of their desire crop area then in the lower righthand corner of the crop area. The program will take those coordinates and make a rectangular area out of those values to then crop the image from. After all desired sections have been selected, close the picture window and the photos will be saved to the folder the original photo came from as well as save the image data to crop_image_data.csv for padding later

The program takes no parameters and can be run by simply using 'python click_crop.py'


## Using auto_crop.py

This file takes in an image file then automatically segments and crops it 

Note: This program is specifically designed for the imaging style of our Field Museum images, it can be used for other imaging but adjustments will need to be made  

When the program is run it asks the user to select a photo they wish to crop. It takes both .jpg and .png files. Once the photo is selected the program will automatically threshold the image and detect the edges of the leaves. It will then display the image with each segment highlighted with a red box. After closing the image window the program will begin to automatically crop, save, and write the image data to the auto_crop_image_data.csv file 

The program takes no parameters and can be run by simply using 'python auto_crop.py'

## Using crop_resize.py

This files uses the image data stored in a CSV file and resizes the images to be as large as the largest on in the file by padding it with white space 

Before runnning the program you need to have a CSV file filled with all the image data for the images you wish to resize. The image data should be stored on separate lines representing length, width, and file path to the image (in that order). When the program is run it will scan through the desired CSV file and find the largest values for both height and width of the image and begin to resize all images stored in the file to that size using white padding. The images are padded using their file path saved in the CSV file


## Using augmentor.py

This file loads in two folders of images, one training set and one testing set, and augments them using the chosen augmentation functions. The user will specify the file path for both training and testing folders as well as the batch size for both. Different augmentation's are written in the code that can be turned on and off to try new combinations of augmentation. This script can be run in the terminal as follows:

python augmentor.py training_image_path, batch_size1, testing_image_path, batch_size2

Example: 
`python augmentor.py training_images 100 testing_images 20`


## Using only_ab.py

This file allows you to search through an image folder and extract images with a certian word in the title. In this case we use it to extract only the abaxial leaf images. The script takes one parameter which is just the file path of the image folder.  

python only_ab.py image_folder_path

Example: 
`python only_ab.py \5_leaf_catagories\catagory1\hippophaeoides`