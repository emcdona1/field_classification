# ml_classifications

The code here creates and tests a CNN model using Tensorflow and Keras that takes images of two morphologically similar plant families (Lycopodieaceae and Selaginellaceae) and trains the model to identify which is which. 

---
# Folder Descriptions

### archive

This folder holds the files to build a model using the data from images stored in pickle files. As of August 27, 2019, it is an older model and would have to be updated to match **build_model_k_fold_cv.py** if you would like to use it. This folder contains 3 files: 
* build_model.py
* input_data.py
* input_data_split.py

Input_data and input_data_split are very similar except that input_data_split exports a pickle file for the features, labels, and image names for training and testing respectively (so a total of 6 pickle files). Input_data.py on the other hand does NOT export separate pickle files for testing and would have to be divided in the build_model file. Currently build_model.py is not written to take in pickle files for both testing and training.

### helpers_or_in_progress

This folder contains files for side projects under the machine learning umbrella and possibly helper functions that may help the user.
A description of the files and folders inside are below.
* **activation_visualization.py** is to create activation maps that can show what each layer is detecting in each image. Currently not running.
* **pick_distributed_images.py** this file 
# File Descriptions
* **build_model.py:** opens data from pickle files, builds model architecture, and trains model. Where most of the change are made
* **image_download.py:** using CSVs from the Pteridophyte Portal, downloads the desired specimen images onto the computer
* **image_resize.py:** takes all images in a folder and resizes them to 256 px by 256 px
* **input_data.py:** holds functions that take the appropriately resized photos and store them as numpy arrays in pickle files. These photos are then used for training/validation by **build_model.py**
* **input_data_split.py:** is a file I was working on for a time
* **build_model_direct_images.py:** I created this file because there was a point when we had too many images to store into pickle files. This python file bypasses that extra steps and directly loads the images into the model. I think it takes a little longer than opening pickle files. Starting on August 20, 2019 and after, all model changes were made on this file.

# Using image_download.py
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

# Using image_resize.py

The purpose of this file is to take the raw downloaded files and convert them into squares of the same size using an image processing package called OpenCV for Python. To install the package, check out [this website](https://pypi.org/project/opencv-python/) 

The program takes in the folder with all the original images, the destination for the resultant images, and the "label" for all these images (for example: the plant family, article of clothing all these images are, etc). In addition to resizing all the images to a default size of 256 x 256 but it can be changed, it outputs a CSV into the destination folder that lists all the images with the corresponding label. 

To use the file, follow this format in the command line terminal:

`python image_resize.py -f orig_image_folder_path -d dest_folder_path -t label_name -s image_size`

For example:
`python image_resize.py -f orig_images -d smaller_images -t cats -s 256`

# input_data.py and input_data_split.py

Both these files have a similar goal, they just differ in how they work. input_data.py takes in two folders of images and will export a 3 pickle files with the images from both folders shuffled together. The pickle files hold the following:

1. The features
2. The labels (ex: class A or class B)
3. Names of the images

The three files will hold the data in the same order. In other words, the 10th array of features corresponds to the the class of the 10th element in the labels files and the 10th name in the names file.

Similarly, input_data_split.py takes in two folders but exports 6 pickle files, 2 groups. 1 group will be used for training data and the other group is used for testing. Which images get sent to which group is done randomly.

If you are using one then switch to using the other, be careful! Half the pickle files from input_data_split.py have the same file names as the exported files from input_data.py.

# build_model.py

This file is where the architecture of the model is created. It imports the pickle files where images and labels are stored, builds the model, and trains it. Here you can change parameters such as the number of epochs, learning rate, regularizers, etc. At the end, the final model is saved and graphs of validation accuracy and loss pop up (1 at a time) so you can get a general idea of the trends of the model on images it does *not* train on.

# test_model.py and test_model_external.py

These two scripts have the same goal of testing the model on a completely new batch of images. The difference is that the test_model.py will test images from the testing pickle files exported by input_data_split.py. On the other hand, test_model_external.py allows you to input 2 folders with the test images manually. Both scripts export a CSV with each image name, the true label, and the predicted label.

# Extra Folders

## frullania

This folder contains adapations of the model that we want to use to train for 2 completely new classes: frullania rostrata and frullania coastal. This application is new to science because there is speculation that these two are different species and hopefully we can train a model to identify morphological differences between the two.

## cross_validation

This folder implements k-fold cross validation to improve the robustness and accuracy of the model.
