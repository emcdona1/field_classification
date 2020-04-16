# field_classification

The code in this repository uses Convolutional Neural Networks (CNN) in Tensorflow/Keras to classify images of two sets of plant species (e.g. the morphologically similar plant families *Lycopodieaceae* and *Selaginellaceae*, or two species of the *Frullania* genus) based on the available corpus of images.  Scripts are available to download and preprocess certain images, and the CNN and classification codes are designed to be generic with multiple types of images and species of plants.  We are currently experimenting with different deep learning architectures, convolution kernels, and how each set of processes work on different corpora of images.


---

## Setup
1. Clone the repository locally.
1. Confirm you have Python and the necessary packages installed (see Environment section below).
1. Prepare two sets of images, each set in a separate directory.  (The directory name indicates the class label, e.g. species.)  Images must be square (1:1 aspect ratio) and of the same dimensions.
    - To resize and reshape images, use `utilities\image_processing\image_resize.py`.  Note that this tool does not crop images, so rectangular images will be skewed (distorting one dimension).
    - To use the Keras built-in CIFAR-10 dataset (reduced to 2 classes), you can manually adjust the `load_image_sets()` method in `train_models_image_classification.py`.
1. If desired, prepare a separate group of test images (not used in training and validation) by using `utilities\image_processing\create_test_group.py`.  By default this program creates a split of 90% for training & validation and 10% for testing.
    - This creates four new directories  -- *folder1test, folder1train, folder2test*, and *folder2train* -- and copies each image file into one of the four new directories.


### Environment
This code has been tested in Python 3.7 using Anaconda (Windows) and Miniconda (Ubuntu).

**Note**: This system is *not* compatible with Tensorflow 2.x (as of April 2020).

#### Tested Package Versions:
- keras v2.3.1
- matplotlib v3.1.3
- numpy v1.18.1
- openCV (cv2) v3.4.2
- pandas v1.0.3
- scikit-learn v0.22.1
- tensorflow v1.15.0

---

## Workflow
- Run `train_models_image_classification.py`, using arguments to specify image sets and hyper-parameters.
    - Arguments: (`-h` flag for full details)
        - `c1` (required) - file path of the first image folder (class 1)
        - `c2` (required) - file path of the second image folder (class 2)
        - (`-color`, `-bw`) - number of color channels (RGB or K) boolean flag
        - `-lr` - learning rate value
        - `f` - number of folds (1 for no cross-fold validation, 2+ for cross-fold validation)
        - `-e` - number of epochs per fold (10+)
        - `-b` - batch size (minimum 2) for updates
    - Output:
        - Directory `saved_models` in current working directory, which contains all generated models (file name format `CNN_#.model`).
        - Directory `graphs` in current working directory, which contains all generated graphs/plots for each run, plus a CSV summary of each fold.
    - Example command: `python train_models_image_classification.py images\species_a images\species_b -f 10 -e 100 -b 64 > species_a_b_training_output.txt`


- After training model(s), classifying images in your test set.  Number of predictions generated = *# of test images * # of model files*
- Run `classify_images_by_vote.py`.
    - Arguments: (`-h` flag for full details)
        - `c1` - file path of the first test image folder (class 1)
        - `c2` - file path of the second image folder (class 2)
        - `m` - folder of generated models (i.e. `saved_models` in working directory)
    - Output:
        - Results are saved as a CSV file (`yyyy-mm-dd-hh-mm-ssmodel_vote_predict.csv`) in `predictions` directory (which is created if needed).  

---

## Repository layout

*** **being restructured** ***

For full details, see section Folder Descriptions below.

- The main files in the root are used for acquiring images, building the model, and training it.
- `archive/` contains an older version of the model and method of uploading images into the model that may still be explored in the future.
- `image_processing/` contains programs for downloading images of the Field Museum's digital collection of the *Lycopodieaceae* and *Selaginellaceae* families from an online database, as well as a tool to resize images.  See the folder for full instructions.



---

## File descriptions
*** **currently being restructured** ***

### train_models_image_classification.py

This file is where the architecture of the model is created and the model is trained and validated. It takes in images directly and performs k-fold cross validation. This file can be called from the terminal with arguments for path to category folders, the category folder names, image size, number of folds, and number of epochs. 

If there are any other parameters that you would like to change in the model, you'd have to directly change the code.

### classify_images_by_vote.py

Given a folder containing multiple models (e.g. from performing k-fold cross validation), this program takes in a folder of model files, generates predictions for each image from each model, and then takes a simple majority vote across all models in order to determine the classification.
This program depends on functions from the classify_images.py program.

---

## Contributors and licensing
This code has been developed by Beth McDonald ([emcdona1](https://github.com/emcdona1), NEIU) and Allison Chen ([allisonchen23](https://github.com/allisonchen23), UCLA), under the guidance of Dr. Matthew Von Konrat (Field Museum), Dr. Francisco Iacobelli ([fiacobelli](https://github.com/fiacobelli), NEIU), Dr. Rachel Trana ([rtrana](https://github.com/rtrana), NEIU), and Dr. Tom Campbell (NEIU).

Code development and testing was made possible thanks to [the Grainger Bioinformatics Center](https://www.fieldmuseum.org/science/labs/grainger-bioinformatics-center) at the Field Museum.  This project has been constructed for the Field Museum Gantz Family Collections Center, under the direction of [Dr. Matthew Von Konrat](https://www.fieldmuseum.org/about/staff/profile/16), Head of Botanical Collections at the Field.  Please contact Dr. Von Konrat for licensing inquiries.