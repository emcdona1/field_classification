# Convolutional Neural Networks and Microplants
#### _field_classification_ repository


The code in this repository uses Convolutional Neural Networks (CNN) in Tensorflow/Keras to classify images of two sets 
of plant species (e.g. the morphologically similar plant families *Lycopodieaceae* and *Selaginellaceae*, or two species 
of the *Frullania* genus) based on the available corpus of images.  Scripts are available to download and preprocess 
images. The CNN and classification programs are generic enough to accept images of any species of plants 
(or other objects!).


---

## Setup
1. Clone the repository to your local machine.
1. Confirm you have the necessary Python version and packages installed (see Environment section below).
1. Prepare two sets of images, each within a directory that indicates the class name.  These folders should be put 
   together in a directory, with no other image files. e.g.,
```
training_image_folder
└───species_a
└───species_b
```
   

4. If you have TIF images, use the script in `utilities\image_process\tif_to_jpg.py` to quickly convert files.  
   (The TIF files will be moved to a new subdirectory called `tif`.)
1. You should prepare a separate group of test images, either manually, or you can use the available utility script: 
   `utilities\image_processing\create_test_group.py`.
    - This script defaults to creates a split of 90% for training/validation and 10% for testing. It creates copies of 
      the images in four new directories  -- *folder1test, folder1train, folder2test*, and *folder2train*.


### Environment
This code has been tested in Python 3.9.4 in Windows and Ubuntu, using Anaconda 
for virtual environments.  Please consult `requirements.txt` or the list below 
for necessary Python packages.

#### Tested Package Versions:
- tensorflow 2.5.0-rc3 (*Release v1.0 and earlier are compatible with TensorFlow 1.15.0*)
- matplotlib~=3.4.2
- numpy==1.19.5
- opencv-python~=4.5.2.52
- pandas==1.2.4
- scikit-learn==0.24.2
- pillow~=8.2.0
- scipy~=1.6.2
- requests~=2.25.1
- scikit-image~=0.18.1
- augmentor~=0.2.8

---

## Workflow
1. Run `train_models_image_classification.py` or `train_handwriting_model.py`, using arguments to specify image sets and hyper-parameters.

- **Arguments**: (`-h` flag for full details)
    - `training_set` (positional, required) - file path of the directory that contains the training images 
      (e.g. `training_image_folder` as described in the Setup section.)
    - `height` (positional, required) - desired image size (if the optional `-w` argument is not provided, images
        will be loaded as `height x height` square)
    - `-w` - image width for non-square images
    - (`-color`, `-bw`) - boolean flag for number of color channels (RGB or K) (*default = color*)
    - `-lr` - learning rate value (*decimal number*, default = 0.001)
    - `-f` - number of folds (1 for single-model, 2+ for cross-fold validation) (*integer <= 1*, default=1)
        - *Note*: Currently, cross-fold validation is not implemented.
    - `-e` - number of epochs per fold (*integer >= 5*, default=25)
    - `-b` - batch size for updates (*integer >= 2*, default=64)
    - `-m` - path for a metadata file containing the labels (this extracts the column named "human_transcription," 
        as created by the zooniverse analysis project)
- **Output**:
    - Directory `saved_models` is created in current working directory, which will contain one model file per fold (file name format: `CNN_#.model`).
    - Directory `graphs` is created in current working directory, which will contain all generated graphs/plots for each run, plus a CSV summary for each fold.
      - Note: This directory will be empty after CTC model training.
- **Example execution (CNN)**: `python train_models_image_classification.py training_images 128 -color -lr 0.005 -f 10 -e 50 -b 64 > species_a_b_training_output.txt &`
  
- **Example execution (CTC)**: `python train_handwriting_model.py testing_image_sets\word_images\ 100 -w 400 -bw -f 1 -e 100 -b 32 -m testing_image_sets\word_images\words_metadata.csv`


2. After the training is finished, use the model file(s) to classify test set images.  The number of predictions generated = *# of test images * # of model files*  To run `classify_images_by_vote.py`:
- **Arguments**: (`-h` flag for full details)
    - `images` (positional, required) - file path of a directory containing the test image folders
    - `img_size` (positional, required) - image size to be used (must match how the model was trained)
    - `models` (positional, required) - a single model file, or a folder of models (i.e. `saved_models` in working directory)
    - (`-color`, `-bw`) - boolean flag for number of color channels (RGB or K) (*default = color*)
- **Output**:
    - Directory `predictions` is created if needed, and the predictions are saved as a CSV file (`yyyy-mm-dd-hh-mm-ssmodel_vote_predict.csv`).
- **Example execution (CNN only)**: `python classify_images_by_vote.py test_images 128 saved_models -color`

---

## Repository layout

##### Folders:

- **_/_** - Contains the main files used for training and testing models.
- **_/data_visualization_** - Contains the files for generating and saving graphs/data visualizations after training. Creates a `graphs` directory, if it doesn't already exist.
- **_/labeled_images_** - Contains the files for loading in image sets.
- **_/models_** - Contains the files used to define neural network layer architectures.
- **_/utilities_** - Contains image preprocessing scripts, a simple program timer, and archived files.

##### Files:

- **train_models_image_classification.py**
    - _The main training program for image classification -- see Workflow above._
- **train_handwriting_model.py**
    - _The main training program for RNN/CTC handwriting digitization -- see Workflow above._
- **classify_images_by_vote.py**
    - _The main testing program for image classification -- see Workflow above._


---

## Contributors and licensing
This code has been developed by Beth McDonald ([emcdona1](https://github.com/emcdona1), *Field Museum, former NEIU*), 
Sean Cullen ([SeanCullen11](https://github.com/SeanCullen11), NEIU)
and Allison Chen ([allisonchen23](https://github.com/allisonchen23), UCLA).

This code was developed under the guidance of Dr. Matt von Konrat (Field Museum), 
Dr. Francisco Iacobelli ([fiacobelli](https://github.com/fiacobelli), *NEIU*), 
Dr. Rachel Trana ([rtrana](https://github.com/rtrana), *NEIU*), 
and Dr. Tom Campbell (*NEIU*).

This project was made possible thanks to [the Grainger Bioinformatics Center](https://www.fieldmuseum.org/science/labs/grainger-bioinformatics-center) at the Field Museum.

This project has been created for use in the Field Museum Gantz Family Collections Center, 
under the direction of [Dr. Matt von Konrat](https://www.fieldmuseum.org/about/staff/profile/16), Head of Botanical Collections at the Field.

Please contact Dr. von Konrat for licensing inquiries.