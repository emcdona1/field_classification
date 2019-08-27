import os
import math
from shutil import copyfile
import argparse


# divide_all will take all the images of class A and put num_training of them in 1 folder and the remaining in another where people can pull from for sampling
# the images are chosen at approximately even intervals to allow for distribution of images
# store_remaining is a boolean. If true, stores remaining images in the remaining path
def divide_all(src_path, num_training, store_remaining, selected_path, remaining_path):
    print("CAUTION!!!")
    print("If you're using this function on a destination folder with images already in it, CLEAR IT.")
    print("Otherwise it just doesn't work out my friend")

    src = os.listdir(src_path)

    num_images = len(src)
    incr_decimal = 1.0*num_images/num_training
    low_skip = math.floor(incr_decimal)
    high_skip = low_skip + 1
    percent_low_skip = 1-round(incr_decimal - low_skip,3) #3 means 3 decimal places
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

if __name__== '__main__':
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-f', '--folder', default= '', help='path to folder with all images')
    parser.add_argument('-d', '--destination_folder', help="destination folder for selected images")
    parser.add_argument('-n', '--number_images', help = "how many images you want")
    parser.add_argument('-k', '--keep_remaining', default= '' , help = 'Keep the remaining images in another folder? EMPTY STRING for false, anything else for true')
    parser.add_argument('-r', '--remaining_folder', default= '', help = 'Folder for remaining images IF k is True')
    args = parser.parse_args()
    if not os.path.exists(args.destination_folder):
        os.makedirs(args.destination_folder)
    if args.keep_remaining and not os.path.exists(args.remaining_folder):
        os.makedirs(args.remaining_folder)
    divide_all(args.folder, int(args.number_images), args.keep_remaining, args.destination_folder, args.remaining_folder)
