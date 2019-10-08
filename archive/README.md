# archive folder

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

### adiantum_blechnum

*Adiantum_blechnum* contains a side project of using the same CNN architecture on two genera (*Adiantum* and *Blechnum*) which are more visually distinct than *Lycopodieaceae* and *Selaginellaceae*).
This was a side branch of the project where we trained the model in this folder to distinguish between two genera that from the human eye already look pretty different. The purpose was to see if our model architecture was even fit for something of this task, and considering this reached 97% accuracy, we think distinguishing between Lycopodium and Selaginella should reach something similar as well. The images this was trained on can be found [here](https://drive.google.com/drive/folders/1Wa58rPV6Z5cbFlN2_Ld9r-8S4OZxiv8l?usp=sharing).

## input_data.py and input_data_split.py

Both these files have a similar goal, they just differ in how they work. input_data.py takes in two folders of images and will export a 3 pickle files with the images from both folders shuffled together. The pickle files hold the following:

1. The features
2. The labels (ex: class A or class B)
3. Names of the images

The three files will hold the data in the same order. In other words, the 10th array of features corresponds to the the class of the 10th element in the labels files and the 10th name in the names file.

Similarly, input_data_split.py takes in two folders but exports 6 pickle files, 2 groups. 1 group will be used for training data and the other group is used for testing. Which images get sent to which group is done randomly.

If you are using one then switch to using the other, be careful! Half the pickle files from input_data_split.py have the same file names as the exported files from input_data.py. Currently build_model.py is not written to take in pickle files for both testing and training.

## list_images_confidence_threshold.py
Abandoned idea of taking a list of images and "flagging" ones below (or above) a certain threshold for manual review.  This was abandoned in favor of changing the threshold (from 0.5 to ~0.74) which fixed the accuracy dramatically.