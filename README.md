# fern_classifications

The code here creates and tests a CNN model using Tensorflow and Keras that takes images of two morphologically similar plant families (Lycopodieaceae and Selaginellaceae) and trains the model to identify which is which. 

# Using image_download.py
The goal of this file is to download images from the Pteridophyte Portal. Before running the file, please follow the steps below:
1. Go to [The Pteridophyte Collections Consortium](http://www.pteridoportal.org/portal/)
2. Click Search > Collections > Deselect All > Choose your source (we chose 'Field Museum of Natural History Pteridophyte Collection')
3. Hit Search in the upper right
4. Fill in your search parameters and hit 'List Display'
5. In the top right, click the little download button (it looks like a down arrow into a open box)
6. Choose the following parameters:

  *Structure: Darwin Core  
   *Data Extensions: Keep both boxes selected   
   *File Format: Comma Delimited (CSV)   
   *Character Set: UTF-8 (Unicode)   
   *Compression: Check this box   
   
7. Hit 'Download Data'

Once you have your downloaded zip file, you will want two CSV's in particular: the images and occurrences
Place these files in the folder that your code is in and create a new folder in this space where you want the images to download.

Run this code by using the following command in terminal/command prompt

`python image_download.py -f [image_csv_name].csv -o [occurrences_csv_name].csv -l [folder_name]`

For example:

`python image_download.py -f images.csv -o occurrences.csv -l specimen_images`

`python image_download.py -f lyco_images.csv -o lyco_occurrences.csv -l lyco_images`

or if the CSVs are in a folder inside the workspace:

`python image_download.py -f lyco_csvs/lyco_images.csv -o lyco_csvs/lyco_occurrences.csv -l lyco_images`

Because of efficiency purposes, the program first looks in the 'identifier' column for the image. If it's not found, it will then look in the 'goodQualityAccessURI' column. If neither produce a useable image, the program will output a CSV with the missing images in the folder you input that lists the barcodes and core id numbers.

# Using image_resize.py

The purpose of this file is to take the raw downloaded files and convert them into squares of the same size using an image processing package called OpenCV for Python. To install the package, check out [this website](https://pypi.org/project/opencv-python/) 

The program takes in the folder with all the original images, the destination for the resultant images, and the "label" for all these images (for example: the plant family, article of clothing all these images are, etc). In addition to resizing all the images to a default size of 256 x 256 but it can be changed, it outputs a CSV into the destination folder that lists all the images with the corresponding label. 

To use the file, follow this format in the command line terminal:

`python image_resize.py -f orig_image_folder_path -d dest_folder_path -t label_name -s image_size`

For example:
`python image_resize.py -f orig_images -d smaller_images -t cats -s 256`

# input_data.py and input_data_split.py

Both these files have a similar goal, they just differ in how they work. input_data.py takes in two folders of images and will export a 3 pickle files with the images from both folders shuffled together. The pickle files hold the following:

1. The features
2. The labels (ex: class A or class B)
3. Names of the images

The three files will hold the data in the same order. In other words, the 10th array of features corresponds to the the class of the 10th element in the labels files and the 10th name in the names file.

Similarly, input_data_split.py takes in two folders but exports 6 pickle files, 2 groups. 1 group will be used for training data and the other group is used for testing. Which images get sent to which group is done randomly.

If you are using one then switch to using the other, be careful! Half the pickle files from input_data_split.py have the same file names as the exported files from input_data.py.

