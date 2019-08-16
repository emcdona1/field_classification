from PIL import Image
from resizeimage import resizeimage
import argparse
import cv2
import os, sys
import pandas as pd

def make_square(file_name, dest_path, dir_path, img_size):
    img = cv2.imread(dir_path + '/' + file_name, -1)
    output_name="N/A"
    try:
        file_name=file_name.replace('.jpg','')
        output_name = file_name+'_rsz.jpg'
        img_rsz = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_AREA)
        cv2.imwrite(dest_path+'/' + output_name, img_rsz)
    except:
        print(file_name + " was unable to be resized")
    return output_name

def resize_folder(dir_path, dest_path, label):
    all_images = os.listdir(dir_path)
    df=pd.DataFrame()
    for i in range(len(all_images)):
        output_name = make_square(all_images[i], dest_path, dir_path)
        df = df.append([[output_name, label]])
        if i % 500 == 0:
            print (i, " images resized")
    df = df.rename({0: 'Image Name', 1: 'Family'}, axis='columns')
    df.to_csv(dest_path + '/_label.csv', encoding= 'utf-8', index=False)

if __name__== '__main__':
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-f', '--folder', help="Path to folder where original images are stored")
    parser.add_argument('-d', '--destination', default='', help='Folder/path where you want image saved')
    parser.add_argument('-t', '--tag', default = '', help="The label for all these images to be in the CSV File")
    parser.add_argument('-s', '--image_size', default=256, help='The new image height/width (assuming output is square)')
    args = parser.parse_args()
    resize_folder(args.folder, args.destination, args.tag, args.image_size)
