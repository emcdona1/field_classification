# field_classification

The code in this repository uses Convolutional Neural Networks (CNN) in Tensorflow/Keras to classify images of two sets of plant species (e.g. the morphologically similar plant families *Lycopodieaceae* and *Selaginellaceae*, or two species of the *Frullania* genus) based on the available corpus of images.  Scripts are available to download and preprocess certain images, and the CNN and classification codes are designed to be generic with multiple types of images and species of plants.  We are currently experimenting with different deep learning architectures, convolution kernels, and how each set of processes work on different corpora of images.
---

## Workflow
- After cloning the repository, gather test images into two folders (named by their classification, e.g. species or family).  Use image processing scripts as needed to reduce images to 256x256.
- Run *create_test_group.py* to split the image sets into two groups: training/validation (90%) and testing (10%).  (This creates four new folders in the same directory, and copies of each file.)
- Run *create_models_with_cross_validation.py* on the training/validation image sets.  (Ed.: We have typically been running 10-fold cross validation on between 100 and 500 epochs.)
- Run *classify_images_by_vote.py* on the test image set folders to gauge the model performance.  Results are saved as a CSV file in \[working-directory\]/predictions/.

---

## Workflow
- After cloning the repository, gather test images into two folders (named by their classification, e.g. species or family).  Use image processing scripts as needed to reduce images to 256x256.
- Run *create_test_group.py* to split the image sets into two groups: training/validation (90%) and testing (10%).  (This creates four new folders in the same directory, and copies of each file.)
- Run *create_models_with_cross_validation.py* on the training/validation image sets.  (Ed.: We have typically been running 10-fold cross validation on between 100 and 500 epochs.)
- Run *classify_images_by_vote.py* on the test image set folders to gauge the model performance.  Results are saved as a CSV file in \[working-directory\]/predictions/.

---

## Repository layout
For full details, see section Folder Descriptions below.

- The main files in the root are used for acquiring images, building the model, and training it.
- *archive/* contains an older version of the model and method of uploading images into the model that may still be explored in the future.
- *image_processing/* contains two programs for downloading and prepping the Field Museum's images of the *Lycopodieaceae* and *Selaginellaceae* families from an online database.  See the folder for full instructions.


### create_models_with_cross_validation.py

This file is where the architecture of the model is created and the model is trained and validated. It takes in images directly (*not* from pickle files) and performs k-fold cross validation. This file can be called from the terminal with arguments for path to category folders, the category folder names, image size, number of folds, and number of epochs. Look at lines 242-249 for the "keys" for each input. 

Note that the first argument (-d or --directory) is the path from current location to the folder **holding the two class folders**. If the class folders are in the current folder, leave it blank!

If there are any other parameters that you would like to change in the model, you'd have to directly change the code.

### classify_images.py

After training model(s), you can use this program to classify folders of images (e.g. for testing).  It classifies images based on which folder they are in, i.e. you must have two folders of images (the output names the class based on the name of the folder), and can only be used to test one model at a time.
The program outputs a CSV file containing the image names, classification, and confusion matrix values.

### classify_images_by_vote.py

Given a folder containing multiple models (e.g. from performing k-fold cross validation), this program takes in a folder of model files, generates predictions for each image from each model, and then takes a simple majority vote across all models in order to determine the classification.
This program depends on functions from the classify_images.py program.

---

## Contributors and licensing
This code base has been built by Allison Chen ([allisonchen23](https://github.com/allisonchen23)) and Beth McDonald ([emcdona1](https://github.com/emcdona1)), under the guidance of Dr. Francisco Iacobelli ([fiacobelli](https://github.com/fiacobelli)), Dr. Matt von Konrat, and Dr. Tom Campbell. This code base has been constructed for the Field Museum Gantz Family Collections Center, under the direction of Dr. Matt von Konrat, Head of Botanical Collections at the Field.  Please contact him for licensing inquiries.
