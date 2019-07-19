#try resizing to a width of 1500 bc that seems to be most common

from PIL import Image
from resizeimage import resizeimage
import argparse
import cv2
import os, sys
# def uniform_width(file_name):
#     with Image.open(file_name, 'r') as img:
#         #img=resizeimage.resize_width(img, 1500)
#         file_name = file_name.replace('.jpg','')
#         img_resized = img.resize((256,256))
#         img_resized.save(file_name+'_rsz.jpg', img_resized.format)

def make_square(file_name, dest_path, dir_path):
    img = cv2.imread(dir_path + '/' + file_name, -1)
    try:
        img_rsz = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
        file_name=file_name.replace('.jpg','')
        cv2.imwrite(dest_path+'/'+file_name+'_rsz.jpg', img_rsz)
    except:
        print(file_name + " was unable to be resized")
    # file_name=file_name.replace('.jpg','')
    # cv2.imwrite(dest_path+'/'+file_name+'_rsz.jpg', img_rsz)

def resize_folder(dir_path, dest_path):
    all_images = os.listdir(dir_path)
    for i in range(len(all_images)):
        # print(all_images[i])
        make_square(all_images[i], dest_path, dir_path)
        if i % 500 == 0:
            print (i, " images resized")

if __name__== '__main__':
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-i', '--image', help='Image to be resized')
    parser.add_argument('-f', '--folder', help="Path to folder where original images are stored")
    parser.add_argument('-d', '--destination', default='', help='Folder/path where you want image saved')
    args = parser.parse_args()
    resize_folder(args.folder, args.destination)
    #make_square(args.image,args.destination)