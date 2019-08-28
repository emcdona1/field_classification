# ml_classifications

The code here creates and tests a CNN model using Tensorflow and Keras that takes images of two morphologically similar plant families (Lycopodieaceae and Selaginellaceae) and trains the model to identify which is which. 

---

# Repository layout
The main files used for acquiring images, building the model, and training it are not in any folders. 'Archive' contains an older version of the model and method of uploading images into the model that may still be explored in the future. 'Helpers_or_in_progress' contains just that: files that may be helpful for specific tasks or work that is still in progress and not yet functional. The in progress files, however, are *not* necessary for the model to run. For more details, scroll down

---

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

# build_model_k_fold_cv.py

This file is where the architecture of the model is created. It takes in images directly (*not* from pickle files) and performs k-fold cross validation. This file can be called from the terminal with arguments for path to category folders, the category folder names, image size, number of folds, and number of epochs. Look at lines 242-249 for the "keys" for each input. 

Note that the first argument (-d or --directory) is the path from current location to the folder **holding the two class folders**. If the class folders are in the current folder, leave it blank!

If there are any other parameters that you would like to change in the model, you'd have to directly change the code.

# test_model.py and test_model_external.py

These two scripts have the same goal of testing the model on a completely new batch of images. The difference is that the test_model.py will test images from the testing pickle files exported by input_data_split.py. On the other hand, test_model_external.py allows you to input 2 folders with the test images manually. Both scripts export a CSV with each image name, the true label, and the predicted label.

# Folder Descriptions

### archive

This folder holds the files to build a model using the data from images stored in pickle files. As of August 27, 2019, it is an older model and would have to be updated to match **build_model_k_fold_cv.py** if you would like to use it. This folder contains 3 files: 
* build_model.py
* input_data.py
* input_data_split.py

# input_data.py and input_data_split.py

Both these files have a similar goal, they just differ in how they work. input_data.py takes in two folders of images and will export a 3 pickle files with the images from both folders shuffled together. The pickle files hold the following:

1. The features
2. The labels (ex: class A or class B)
3. Names of the images

The three files will hold the data in the same order. In other words, the 10th array of features corresponds to the the class of the 10th element in the labels files and the 10th name in the names file.

Similarly, input_data_split.py takes in two folders but exports 6 pickle files, 2 groups. 1 group will be used for training data and the other group is used for testing. Which images get sent to which group is done randomly.

If you are using one then switch to using the other, be careful! Half the pickle files from input_data_split.py have the same file names as the exported files from input_data.py. Currently build_model.py is not written to take in pickle files for both testing and training.

---

### helpers_or_in_progress

This folder contains files for side projects under the machine learning umbrella and possibly helper functions that may help the user.
A description of the files and folders inside are below.
* **activation_visualization.py (unfinished)** is to create activation maps that can show what each layer is detecting in each image. Currently not running.
* **pick_distributed_images.py** this file allows users to select images from a folder that are spread out. For example, if your folder has 500 images, but you only want 300, using this python file will select 300 images that are approximately evenly spaced out in the folder. It helps ensure that your resulting selection of 300 is (hopefully) representative of the whole. Additionally, you can choose whether or not to keep the remaining (in this case, 200) images in a separate folder or not. The original folder remains untouched.
* **resume_training.py** allows user to load a model, the features and labels pickle files and resume training at the epoch they left off at. Honestly, I didn't use this file that much and didn't bother creating an argument parse. If you want to change anything with the number of epochs, edit it on line 14. The initial epoch should actually be the epoch you want to start at - 1. So if last training completed 20 epochs, set initial_epoch = 20. The epochs = ## is the total number of epochs that the model will have trained. So if initial_epochs = 20 and epochs = 30, the model will train for 10 more epochs and save (*not* 30 more epochs).
* **roc_crossfold_example.py** is a very simple ML model where we were able to get code from regarding plotting ROC curves at the end of all the folds. (Thank you Beth McDonald for finding this and helping implement!)
* **test_model.py** tests the model on images that are already saved in pickle files. To be used with **build_model.py** in the archive folder. Because this file is not as popular, there is also **no** argument parser. To change the pickle files, edit lines 14, 16, 17. to change where the model is being loaded from, edit the path in line 21.
* **test_model_external.py** takes in two folders and will test the model on these images. Eliminates need for pickle files.

#### Folders in helpers_or_in_progress

##### frullania

This folder contains adapations of the model that we want to use to train for 2 completely new classes: frullania rostrata and frullania coastal. This application is new to science because there is speculation that these two are different species and hopefully we can train a model to identify morphological differences between the two.
Files:
* **frullania_model.py** currently takes images directly and uses bootstrapping to select the training and testing. Will train the model 10 times, to mimic the 10 folds of k-fold cross validation, but the images used are random each time. It currently isn't able to generalize the learning and changes to the kernels of the model would have to be made.
* **img_preprocessing.py(in progress)** started this to not only resize the images but also to use image augmentation to provide greater diversity in the images for training (such as reflecting horizontally or adjusting brightness). Currently not up and running
* **remove_tif_and_duplicates.py** This file was created because when I received the images, some where .jpg and some where .tif. I used imageMagick to convert the .tif files to .jpg, but this package saves the .tif files and ends up making 2 .jpg files. This script simply removes the extraneous files and renames the remaining files appropriately. Really is for a pretty niche problem I faced.

##### cross_validation

This folder implements k-fold cross validation to improve the robustness and accuracy of the model.
