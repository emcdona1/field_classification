# Convolutional Neural Networks and Microplants
#### _field_classification_ repository


The code in this repository uses Convolutional Neural Networks (CNN) in Tensorflow/Keras to classify images of two sets of plant species (e.g. the morphologically similar plant families *Lycopodieaceae* and *Selaginellaceae*, or two species of the *Frullania* genus) based on the available corpus of images.  Scripts are available to download and preprocess images. The CNN and classification programs are designed to accept any standard image file type (per OpenCV standards), and is generic enough to accept images of any species of plants (or other objects!).


---

## Setup
1. Clone the repository to your local machine.
1. Confirm you have Python and the necessary packages installed (see Environment section below).
1. Prepare two sets of images, with each set in a separate directory.  (The directory names will be used as the class label, e.g. *cat* and *dog*.)  Images must be square (1:1 aspect ratio) and all of the same dimensions.
    - To resize and reshape images, use `utilities\image_processing\image_resize.py`.  Note that rectangular images will be skewed (distorting one dimension) to avoid cropping out useful information.
    - Instead of providing your own image sets, a modification of the CIFAR-10 built-in data set is available. To use these images, you will need to manually adjust the `load_image_sets()` method in `train_models_image_classification.py`.
1. If desired, you should prepare a separate group of test images (i.e. images not used in training and validation).  You can do so manually, or via `utilities\image_processing\create_test_group.py`.
    - By default this script creates a split of 90% for training & validation and 10% for testing, and creates four new directories  -- *folder1test, folder1train, folder2test*, and *folder2train*. The script creates new copies of each image file into one of the four directories (the original image files are retained).


### Environment
This code has been tested in Python 3.7 using Anaconda (Windows) and Miniconda (Ubuntu).


#### Tested Package Versions:
See also `requirements.txt`
- keras v2.3.1
- matplotlib v3.1.3
- numpy v1.18.1
- openCV (cv2) v3.4.2
- pandas v1.0.3
- tensorflow v1.15.0  (**N.B.**: *This system is not yet compatible with Tensorflow 2.x.*)

---

## Workflow
- Run `train_models_image_classification.py`, using arguments to specify image sets and hyper-parameters.
    - Arguments: (`-h` flag for full details)
        - `c1` (positional, required) - file path of the first image folder (class 1 training images)
        - `c2` (positional, required) - file path of the second image folder (class 2 trianing images)
        - (`-color`, `-bw`) - boolean flag for number of color channels (RGB or K) (*default = color*)
        - `-lr` - learning rate value (*decimal number*)
        - `f` - number of folds (1 for no cross-fold validation, 2+ for cross-fold validation) (*integer <= 1*)
        - `-e` - number of epochs per fold (*integer >= 10*)
        - `-b` - batch size for updates (*integer >= 2*)
    - Output:
        - Directory `saved_models` created in current working directory, which contains all generated models (file name format: `CNN_#.model`).
        - Directory `graphs` created in current working directory, which contains all generated graphs/plots for each run, plus a CSV summary for each cross-validation fold.
    - Example command: `python train_models_image_classification.py images\species_a images\species_b -f 10 -e 100 -b 64 > species_a_b_training_output.txt`


- After the training is finished, use the model file(s) to classify test set images.  The number of predictions generated = *# of test images * # of model files*
- Run `classify_images_by_vote.py`.
    - Arguments: (`-h` flag for full details)
        - `c1` - file path of the first test image folder (class 1)
        - `c2` - file path of the second image folder (class 2)
        - `m` - folder of generated models (i.e. `saved_models` in working directory)
    - Output:
        - Results are saved as a CSV file (`yyyy-mm-dd-hh-mm-ssmodel_vote_predict.csv`) in `predictions` directory (which is created if needed).  
    - Example command: `python classify_images_by_vote.py images\species_a_test images\species_b_test saved_models\`

---

## Repository layout

##### Folders:
- **_/_** - Contains the main files used for training and testing models.
- **_/labeled_images_** - Contains the files for loading in image sets. 
- **_/models_** - Contains the files used to define neural network layer architectures.
- **_/utilities_** - Contains image preprocessing scripts, a simple program timer, and archived files.

##### Files:

- **classify_images_by_vote.py**
    - _The main testing program -- see Workflow above._
- **cnnarguments.py**
    - _Parse and validate command-line argument values._
- **data_and_visualization_io.py**
    - _Used to create `graphs` directory during model training (see Workflow above)._
- **model_training.py**
    - _A helper class for `train_models_image_classification.py` which takes in two preprocessed image sets and a defined CNN architecture, and trains/validates a model._
- **train_models_image_classification.py**
    - _The main training program -- see Workflow above._



---

## Contributors and licensing
This code has been developed by Beth McDonald ([emcdona1](https://github.com/emcdona1), NEIU) and Allison Chen ([allisonchen23](https://github.com/allisonchen23), UCLA), under the guidance of Dr. Matthew Von Konrat (Field Museum), Dr. Francisco Iacobelli ([fiacobelli](https://github.com/fiacobelli), NEIU), Dr. Rachel Trana ([rtrana](https://github.com/rtrana), NEIU), and Dr. Tom Campbell (NEIU).

Code development and testing was made possible thanks to [the Grainger Bioinformatics Center](https://www.fieldmuseum.org/science/labs/grainger-bioinformatics-center) at the Field Museum.  This project has been constructed for the Field Museum Gantz Family Collections Center, under the direction of [Dr. Matthew Von Konrat](https://www.fieldmuseum.org/about/staff/profile/16), Head of Botanical Collections at the Field.  Please contact Dr. Von Konrat for licensing inquiries.
