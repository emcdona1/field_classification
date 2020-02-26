# field_classification

The code in this repository uses Convolutional Neural Networks (CNN) in Tensorflow/Keras to classify images of two sets of plant species (e.g. the morphologically similar plant families *Lycopodieaceae* and *Selaginellaceae*, or two species of the *Frullania* genus) based on the available corpus of images.  Scripts are available to download and preprocess certain images, and the CNN and classification codes are designed to be generic with multiple types of images and species of plants.  We are currently experimenting with different deep learning architectures, convolution kernels, and how each set of processes work on different corpora of images.


---

## Workflow
- Clone the repository locally.
- From the `utilities/preparation/` directory:
    - Gather test images into two folders (one folder per species or family of plants, e.g. `./coastal/` and `./rostrata/`).  Images must be square for this architecture -- use `image_resize.py` to prepare images (note that this utility skews images, rather than crop).
    - Run `create_test_group.py` to split the image sets into two groups: training/validation (90%) and testing (10%).  (This creates four new folders -- `folder1test`, `folder1train`, `folder2test`, and `folder2train` -- and new copies of each image file.) Argument flags:
      + `-d` : directory of the image folders (optional, defaults to current working directory)
      + `-c1` : folder name of the category 1 images
      + `-c2` : folder name of the category 2 images
      + `-v` : verbose mode (1 = on, 0 = off)


- Run `create_models_with_cross_validation.py` on the training/validation image sets.
    - Execute `python create_models_with_cross_validation.py -h` for a list of required arguments.
    - Environment package dependencies: `tensorflow`, `numpy`, `sklearn`, `matplotlib`
    - Local package dependencies: `models` and `labeled_images`
    - Local file dependencies: `timer.py`, `data_and_visualization_io.py`, and `model_training.py`
    - ** This program is currently being restructured. **


- Run `classify_images_by_vote.py` on the test image set folders to gauge the model performance.  Results are saved as a CSV file in `/predictions/`. Argument flags:
    - Execute `python classify_images_by_vote.py -h` for a list of required arguments.
    - ** This program is currently being restructured. **
  

---

## Repository layout
** currently being restructured **

For full details, see section Folder Descriptions below.

- The main files in the root are used for acquiring images, building the model, and training it.
- `archive/` contains an older version of the model and method of uploading images into the model that may still be explored in the future.
- `image_processing/` contains programs for downloading images of the Field Museum's digital collection of the *Lycopodieaceae* and *Selaginellaceae* families from an online database, as well as a tool to resize images.  See the folder for full instructions.



---

## File descriptions
** currently being restructured **

### create_models_with_cross_validation.py

This file is where the architecture of the model is created and the model is trained and validated. It takes in images directly and performs k-fold cross validation. This file can be called from the terminal with arguments for path to category folders, the category folder names, image size, number of folds, and number of epochs. 

Note that the first argument (-d or --directory) is the path from current location to the folder **holding the two class folders**. If the class folders are in the current working directory, leave it blank!

If there are any other parameters that you would like to change in the model, you'd have to directly change the code.

### classify_images.py

After training model(s), you can use this program to classify folders of images (e.g. for testing).  It classifies images based on which folder they are in, i.e. you must have two folders of images (the output names the class based on the name of the folder), and can only be used to test one model at a time.
The program outputs a CSV file containing the image names, classification, and confusion matrix values.

### classify_images_by_vote.py

Given a folder containing multiple models (e.g. from performing k-fold cross validation), this program takes in a folder of model files, generates predictions for each image from each model, and then takes a simple majority vote across all models in order to determine the classification.
This program depends on functions from the classify_images.py program.

---

## Contributors and licensing
This code base has been developed by Beth McDonald ([emcdona1](https://github.com/emcdona1)) and Allison Chen ([allisonchen23](https://github.com/allisonchen23)), under the guidance of Dr. Francisco Iacobelli ([fiacobelli](https://github.com/fiacobelli)), Dr. Matt von Konrat, Dr. Rachel Trana([rtrana](https://github.com/rtrana)), and Dr. Tom Campbell.
This code has been constructed for the Field Museum Gantz Family Collections Center, under the direction of Dr. Matt von Konrat, Head of Botanical Collections at the Field.  Please contact him for licensing inquiries.

## Software requirements
All code is written in Python 3 and has been tested in Python 3.4.3 and 3.7.0.