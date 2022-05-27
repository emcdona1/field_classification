from matplotlib import image
import cv2
import os
from PIL import Image
import csv
import numpy as np
from pathlib import Path
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import filters, segmentation, measure, morphology, color

BACKGROUND_COLOR_THRESHOLD = 200

# Local variables to help set the x and y axies and declare the list
x1 = 0
y1 = 0
lst = []


def main(image_filename: Path, image_save_path: Path):
    thresholded_image_save_location = img_thresh(image_filename, image_save_path)
    thresholded_image = image.imread(thresholded_image_save_location)
    segment_image(thresholded_image, image_filename, image_save_path)


def segment_image(img, image_filename: Path, image_save_path: Path):
    thresh = filters.threshold_otsu(img)
    bw = morphology.closing(img > thresh, morphology.square(5))
    cleared = segmentation.clear_border(bw)
    label_image = measure.label(cleared)

    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = color.label2rgb(label_image, image=img, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for region in measure.regionprops(label_image):
        if region.area >= 8000:
            # Set the length and height of each box to help filer out unwanted segments
            minr, minc, maxr, maxc = region.bbox
            length = (maxr - minr)
            width = (maxc - minc)

            # Only draw boxes around segments to the right 1000 pixels and that are wider than long
            # This helps us not segment out parts of the ruler in or image that mess with our data
            if minc > 1000 and length < width:
                # draw rectangle around segmented leaves
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

                lst.append(minc)
                lst.append(minr)
                lst.append(maxc)
                lst.append(maxr)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    crop(image_filename, image_save_path)


def crop(image_filename: Path, image_save_path: Path):
    count = 0
    runs = 0

    for item in lst:
        while count < (len(lst) - 3):
            x1 = lst[count]
            x2 = lst[count + 1]
            y1 = lst[count + 2]
            y2 = lst[count + 3]

            # Open original image and crop using our
            # coordinates we selected with slight padding
            img_crop = Image.open(image_filename)
            img_crop_res = img_crop.crop((x1 - 10, x2 - 10, y1 + 10, y2 + 10))

            species = image_filename.parent.stem
            file = image_filename.stem

            count += 4
            runs += 1

            saved_file = f'{species}_{file}_AUTOCROP_{runs}.jpg'
            img_crop_res.save(Path(image_save_path, saved_file))
            print(f'Saved new image: {saved_file}.')

            imageinfo = np.asarray(img_crop_res)
            height = imageinfo.shape[0]
            width = imageinfo.shape[1]

            file_location = Path(image_save_path, saved_file)
            with open(Path(image_save_path, 'auto_crop_image_data.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(([height, width, file_location]))


def img_thresh(image_filename: Path, image_save_path: Path) -> Path:
    img = cv2.imread(str(image_filename), cv2.IMREAD_GRAYSCALE)
    ret, thresh2 = cv2.threshold(img, BACKGROUND_COLOR_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    species = image_filename.parent.stem
    file = image_filename.stem
    tmp_folder = Path(image_save_path, 'tmp')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    image_name = f'{species}_{file}_THRESH.jpg'
    threshold_save_path = Path(tmp_folder, image_name)
    cv2.imwrite(str(threshold_save_path), thresh2)
    return threshold_save_path


if __name__ == '__main__':
    filename = Path(filedialog.askopenfilename())
    save_path = Path(filename.parent, 'auto_cropped_images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main(filename, save_path)
