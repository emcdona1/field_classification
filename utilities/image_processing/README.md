# Convolutional Neural Networks and Microplants
#### _utilities\image_processing_ directory

The code here creates and tests a CNN model using Tensorflow and Keras that takes images of two morphologically 
similar plant families (*Lycopodieaceae* and *Selaginellaceae*) and trains the model to identify which is which. 

---

## Using image_download.py
**Note**: The file `Download_Image_Files_from_Pteridoportal.ipynb` is a standalone version 
of this program which is designed to be run in the browser by non-developers.

The goal of this file is to download images from the Pteridophyte Portal. Before running the file, please follow the 
steps below:
1. Go to [The Pteridophyte Collections Consortium](http://www.pteridoportal.org/portal/)
2. Click Search > Collections > Deselect All > Choose your source (we chose 'Field Museum of Natural History 
   Pteridophyte Collection')
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
Place these files in the folder that your code is in and create a new folder in this space where you want 
the images to download.

Run this code by using the following command in terminal/command prompt

`python image_download.py -f [image_csv_name].csv -o [occurrences_csv_name].csv -l [folder_name]`

For example:

`python image_download.py -f images.csv -o occurrences.csv -l specimen_images`

`python image_download.py -f lyco_images.csv -o lyco_occurrences.csv -l lyco_images`

or if the CSVs are in a folder inside the workspace:

`python image_download.py -f lyco_csvs/lyco_images.csv -o lyco_csvs/lyco_occurrences.csv -l lyco_images`

For efficiency purposes, the program first looks in the 'identifier' column of the first csv for the image. 
If it's not found, it will then look in the 'goodQualityAccessURI' column. If neither produce a usable image, 
the program will output a CSV with the missing images in the folder you input that lists the barcodes and 
core id numbers.

---

## Using image_resize.py

The purpose of this file is to take the raw downloaded files and convert them into squares of the same size using an image processing package called OpenCV for Python. To install the package, check out [this website](https://pypi.org/project/opencv-python/) 

The program takes in the folder with all the original images, the destination for the resultant images, and the desired image size (just as a number because it's resizing to a square). The default size is 256 x 256 pixels, but it can be changed.

To use the file, follow this format in the command line terminal:

`python image_resize.py -sf orig_image_folder_path -df dest_folder_path -s image_size`

For example:
`python image_resize.py -sf orig_images -df smaller_images -s 256`

---
