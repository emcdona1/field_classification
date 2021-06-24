import cv2
import numpy as np
import csv


def main():
    resize()


# Function to find the largest dimensions stored in the CSV file
def find_new_dimensions():
    # Instance variables to store max values
    max_height = 0
    max_width = 0

    # Open CSV file and read stored value to find the largest values
    with open('cropped_image_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Ensure row is not null
            if row:
                height = int(row[0])
                width = int(row[1])
                if height > max_height:
                    max_height = height
                if width > max_width:
                    max_width = width
    return max_height, max_width


# Function to resize all the images in the CSV file
#  to the desired canvas size
def resize():
    # Call find_new_dimensions and store the new
    # canvas width and height
    height, width = find_new_dimensions()

    # Counter to save images
    count = 0

    # Open and read CSV file
    with open('cropped_image_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Ensure row is not null
            if row:
                path = str(row[2])

                # read image
                img = cv2.imread(path)

                # Store img height, width, and color
                ht, wd, cc = img.shape

                # create new image of desired size and color for padding
                ww = width
                hh = height
                color = (252, 248, 245)
                result = np.full((hh, ww, cc), color, dtype=np.uint8)

                # Find center of canvas to place the original image
                xx = (ww - wd) // 2
                yy = (hh - ht) // 2

                # copy image into center of canvas
                result[yy:yy + ht, xx:xx + wd] = img

                # Increment count
                count += 1

                # Split our filename to get the desired sections
                # and exclude the .jpg or .png extension
                path_split = path.split("/")
                file = path_split[-1].replace('.jpg', '')
                file = file.replace('.png', '')

                # Merge back together our file name excluding the
                # parts we do not want
                a = '/'
                path = path_split[:-1]
                path = a.join(path)
                print(path)

                # save result
                cv2.imwrite(path + '/' + "padded_" + file + ".jpg", result)
                print('saved ' + str(count))


# Call main function
if __name__ == "__main__":
    main()
