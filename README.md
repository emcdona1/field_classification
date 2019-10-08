# field_classification

The code here creates and tests a CNN model using Tensorflow and Keras that takes images of two morphologically similar plant families (*Lycopodieaceae* and *Selaginellaceae*) and trains the model to identify which is which. 

---

## Repository layout
For full details, see section Folder Descriptions below.

- The main files in the root are used for acquiring images, building the model, and training it.
- *archive/* contains an older version of the model and method of uploading images into the model that may still be explored in the future.
- *image_processing/* contains two programs for downloading and prepping the Field Museum's images of the *Lycopodieaceae* and *Selaginellaceae* families from an online database.  See the folder for full instructions.

---


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