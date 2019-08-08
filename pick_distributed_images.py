import os
import math
from shutil import copyfile

root="data"
# divide_all will take all the images of class A and put num_training of them in 1 folder and the remaining in another where people can pull from for sampling
# the images are chosen at approximately even intervals to allow for distribution of images
def divide_all(folder_name, num_training):
    src_path = os.path.join(root, folder_name)
    src = os.listdir(src_path)
    selected_path = os.path.join(root,folder_name+"_train_selected") 
    remaining_path = os.path.join(root,folder_name+"_test_selected") 
    num_images = len(src)
    incr_decimal = 1.0*num_images/num_training
    low_skip = math.floor(incr_decimal)
    high_skip = low_skip + 1
    percent_low_skip = 1-round(incr_decimal - low_skip,3)
    num_low_skip = math.floor(percent_low_skip*num_images)
    i = 1
    cur_skip = low_skip
    for image in src:
        # every image gets saved to one folder or another
        if i%cur_skip == 0:
            copyfile(os.path.join(src_path,image), os.path.join(selected_path, image))
        else:
            copyfile(os.path.join(src_path,image), os.path.join(remaining_path, image))

        if i > num_low_skip:
            cur_skip = high_skip
        i+=1

# divide_all("Lycopodiaceae",1000)
divide_all("Selaginellaceae",1000)