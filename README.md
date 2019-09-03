# field_classification

The code here creates and tests a CNN model using Tensorflow and Keras that takes images of two morphologically similar plant families (*Lycopodieaceae* and *Selaginellaceae*) and trains the model to identify which is which. 

---

# Repository layout
For full details, see section Folder Descriptions below.

- The main files in the root are used for acquiring images, building the model, and training it.
- *Adiantum_blechnum* contains a side project of using the same CNN architecture on two genera (*Adiantum* and *Blechnum*) which are more visually distinct than *Lycopodieaceae* and *Selaginellaceae*).
- *Archive/* contains an older version of the model and method of uploading images into the model that may still be explored in the future.
- *Helpers_or_in_progress/* contains files that may be helpful for specific tasks or work that is still in progress and not yet functional.

---
# Downloading and processing images, and training the model

## Using image_download.py
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

`python image_resize.py -f orig_image_folder_path -d dest_folder_path -s image_size`

For example:
`python image_resize.py -f orig_images -d smaller_images -s 256`

## build_model_k_fold_cv.py

This file is where the architecture of the model is created. It takes in images directly (*not* from pickle files) and performs k-fold cross validation. This file can be called from the terminal with arguments for path to category folders, the category folder names, image size, number of folds, and number of epochs. Look at lines 242-249 for the "keys" for each input. 

Note that the first argument (-d or --directory) is the path from current location to the folder **holding the two class folders**. If the class folders are in the current folder, leave it blank!

If there are any other parameters that you would like to change in the model, you'd have to directly change the code.

# Folder Descriptions

### adiantum_blechnum

This was a side branch of the project where we trained the model in this folder to distinguish between two genera that from the human eye already look pretty different. The purpose was to see if our model architecture was even fit for something of this task, and considering this reached 97% accuracy, we think distinguishing between Lycopodium and Selaginella should reach something similar as well. The images this was trained on can be found [here](https://drive.google.com/drive/folders/1Wa58rPV6Z5cbFlN2_Ld9r-8S4OZxiv8l?usp=sharing).

### archive

This folder holds the files to build a model using the data from images stored in pickle files. As of August 27, 2019, it is an older model and would have to be updated to match **build_model_k_fold_cv.py** if you would like to use it. This folder contains 3 files: 
* build_model.py
* input_data.py
* input_data_split.py

### helpers_or_in_progress

This folder contains files for side projects under the machine learning umbrella and possibly helper functions that may help the user.
A description of the files and folders inside are below.
* **activation_visualization.py (unfinished)** is to create activation maps that can show what each layer is detecting in each image. Currently not running.
* **confusion_matrix.py (unfinished)** creates a confusion matrix. Could be formatted a lot better
* **pick_distributed_images.py** this file allows users to select images from a folder that are spread out. For example, if your folder has 500 images, but you only want 300, using this python file will select 300 images that are approximately evenly spaced out in the folder. It helps ensure that your resulting selection of 300 is (hopefully) representative of the whole. Additionally, you can choose whether or not to keep the remaining (in this case, 200) images in a separate folder or not. The original folder remains untouched.
* **resume_training.py** allows user to load a model, the features and labels pickle files and resume training at the epoch they left off at. Honestly, I didn't use this file that much and didn't bother creating an argument parse. If you want to change anything with the number of epochs, edit it on line 14. The initial epoch should actually be the epoch you want to start at - 1. So if last training completed 20 epochs, set initial_epoch = 20. The epochs = ## is the total number of epochs that the model will have trained. So if initial_epochs = 20 and epochs = 30, the model will train for 10 more epochs and save (*not* 30 more epochs).
* **roc_crossfold_example.py** is a very simple ML model where we were able to get code from regarding plotting ROC curves at the end of all the folds. (Thank you Beth McDonald for finding this and helping implement!)
* **test_model.py** tests the model on images that are already saved in pickle files. To be used with **build_model.py** in the archive folder. Because this file is not as popular, there is also **no** argument parser. To change the pickle files, edit lines 14, 16, 17. to change where the model is being loaded from, edit the path in line 21.
* **test_model_external.py** takes in two folders and will test the model on these images. Eliminates need for pickle files.
* **test_cv_model_external.py** similar to test_model_external.py but instead of telling the file the model, you tell the file where the *folder* of models is. It will go through the folder, load all the models, and test them on the folders of images you feed in. Good because cross validated models actually consist of more than one model and you want to take them all into consideration. Uses the probabilities outputted by the final layer.

---

## Contributors and licensing
This code base has been built by Allison Chen ([allisonchen23](https://github.com/allisonchen23)) and Beth McDonald ([emcdona1](https://github.com/emcdona1)), under the guidance of Dr. Francisco Iacobelli ([fiacobelli](https://github.com/fiacobelli)), Dr. Matt von Konrat, and Dr. Tom Campbell. This code base has been constructed for the Field Museum Gantz Family Collections Center, under the direction of Dr. Matt von Konrat, Head of Botanical Collections at the Field.  Please contact him for licensing inquiries.