# field_classification

The code here creates and tests a CNN model using Tensorflow and Keras that takes images of two morphologically similar plant families (*Lycopodieaceae* and *Selaginellaceae*) and trains the model to identify which is which. 

---

# Repository layout
For full details, see section Folder Descriptions below.

- The main files in the root are used for acquiring images, building the model, and training it.
- *archive/* contains an older version of the model and method of uploading images into the model that may still be explored in the future.
- *image_processing/* contains two programs for downloading and prepping the Field Museum's images of the *Lycopodieaceae* and *Selaginellaceae* families from an online database.  See the folder for full instructions.

---


## build_model_k_fold_cv.py

This file is where the architecture of the model is created. It takes in images directly (*not* from pickle files) and performs k-fold cross validation. This file can be called from the terminal with arguments for path to category folders, the category folder names, image size, number of folds, and number of epochs. Look at lines 242-249 for the "keys" for each input. 

Note that the first argument (-d or --directory) is the path from current location to the folder **holding the two class folders**. If the class folders are in the current folder, leave it blank!

If there are any other parameters that you would like to change in the model, you'd have to directly change the code.

---

## Contributors and licensing
This code base has been built by Allison Chen ([allisonchen23](https://github.com/allisonchen23)) and Beth McDonald ([emcdona1](https://github.com/emcdona1)), under the guidance of Dr. Francisco Iacobelli ([fiacobelli](https://github.com/fiacobelli)), Dr. Matt von Konrat, and Dr. Tom Campbell. This code base has been constructed for the Field Museum Gantz Family Collections Center, under the direction of Dr. Matt von Konrat, Head of Botanical Collections at the Field.  Please contact him for licensing inquiries.