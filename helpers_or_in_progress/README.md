# helpers_or_in_progress folder

Information about sub folders:

### frullania folder

This folder contains adapations of the model that we want to use to train for 2 completely new classes: frullania rostrata and frullania coastal. This application is new to science because there is speculation that these two are different species and hopefully we can train a model to identify morphological differences between the two.
Files:
* **bootstrapping_frullania_model.py** currently takes images directly and uses bootstrapping to select the training and testing. Will train the model 10 times, to mimic the 10 folds of k-fold cross validation, but the images used are random each time. It currently isn't able to generalize the learning and changes to the kernels of the model would have to be made.
* **img_preprocessing.py(in progress)** started this to not only resize the images but also to use image augmentation to provide greater diversity in the images for training (such as reflecting horizontally or adjusting brightness). Currently not up and running
* **remove_tif_and_duplicates.py** This file was created because when I received the images, some where .jpg and some where .tif. I used imageMagick to convert the .tif files to .jpg, but this package saves the .tif files and ends up making 2 .jpg files. This script simply removes the extraneous files and renames the remaining files appropriately. Really is for a pretty niche problem I faced.

### cross_validation folder

This folder implements k-fold cross validation to improve the robustness and accuracy of the model.