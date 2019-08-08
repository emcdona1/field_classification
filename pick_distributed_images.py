import os
import math
from shutil import copyfile

root="data"
# divide_all will take all the images of class A and put num_training of them in 1 folder and the remaining in another where people can pull from for sampling
# the images are chosen at approximately even intervals to allow for distribution of images
# store_remaining is a boolean. If true, stores remaining images in the remaining path
def divide_all(folder_name, num_training, store_remaining):
    print("CAUTION!!!")
    print("If you're using this function on a destination folder with images already in it, CLEAR IT.")
    print("Otherwise it just doesn't work out my friend")
    src_path = os.path.join(root, folder_name)
    src = os.listdir(src_path)
    # Use following paths when choosing training data
    # selected_path = os.path.join(root,folder_name+"_selected") 
    # remaining_path = os.path.join(root,folder_name+"_remaining") 

    # Use following paths for test data (store_remaining should be set to FALSE)
    selected_path = os.path.join(root,folder_name+"_testing")
    remaining_path = ""

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
            if store_remaining:
                copyfile(os.path.join(src_path,image), os.path.join(remaining_path, image))

        if i > num_low_skip:
            cur_skip = high_skip
        i+=1

# divide_all("Lycopodiaceae", 800, True)
# divide_all("Selaginellaceae", 800, True)
divide_all("Selaginellaceae_remaining", 200, False)
divide_all("Lycopodiaceae_remaining", 200, False)
