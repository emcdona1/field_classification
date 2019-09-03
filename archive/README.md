# archive folder

## input_data.py and input_data_split.py

Both these files have a similar goal, they just differ in how they work. input_data.py takes in two folders of images and will export a 3 pickle files with the images from both folders shuffled together. The pickle files hold the following:

1. The features
2. The labels (ex: class A or class B)
3. Names of the images

The three files will hold the data in the same order. In other words, the 10th array of features corresponds to the the class of the 10th element in the labels files and the 10th name in the names file.

Similarly, input_data_split.py takes in two folders but exports 6 pickle files, 2 groups. 1 group will be used for training data and the other group is used for testing. Which images get sent to which group is done randomly.

If you are using one then switch to using the other, be careful! Half the pickle files from input_data_split.py have the same file names as the exported files from input_data.py. Currently build_model.py is not written to take in pickle files for both testing and training.