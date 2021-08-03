import cv2
from PIL import Image
from tkinter import filedialog
import numpy
import csv
import sys

# Local variables to help set the x and y axies and declare the list
x1 = 0
y1 = 0
list = []


def main():

    # displaying the image in defined window
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('image', img)
    cv2.resizeWindow('image', 1400, 800)

    # Calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # Let user close window
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


# The function to run when the left mouse is clicked
def click_event(event, x, y, flags, params):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:

        # Print coordinates of the mouse click and store those values
        # in our x and y variables and add them to the list
        x1 = int(x)
        y1 = int(y)
        list.append(x1)
        list.append(y1)

        # Show coordinates on mouse click
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

        # Only crop the image once all 4 coordinates
        # have been entered into the list
        if len(list) > 3:
            left = list[0]
            top = list[1]
            right = list[2]
            bottom = list[3]

            # Open original image and crop using our
            # coordinates we selected
            img_crop = Image.open(filename)
            img_crop_res = img_crop.crop((left, top, right, bottom))

            # clear list to crop again
            list.clear()

            # Increment count to keep track of the
            # number of leaves we crop
            count += 1

            # Split our filename to get the desired sections
            # and exclude the .jpg or .png extension
            fn_split = filename.split("/")
            species = fn_split[-2]
            file = fn_split[-1].replace('.jpg', '')
            file = file.replace('.png', '')

            # Merge back together our file name excluding the
            # parts we do not want
            a = '/'
            path = fn_split[:-1]
            path = a.join(path)
            print(path)

            # Save cropped image and cropped image file name
            saved_file = species + '_' + file + '_' + str(count) + '.jpg'
            img_crop_res.save(path + "/" + species + '_' + file + '_' + str(count) + '.jpg')

            # Print the file name and where it was saved
            print('Saved File: ' + saved_file + ' to ' + path)

            # convert image object into array
            imageinfo = numpy.asarray(img_crop_res)

            # Store the image height and width and new file location
            height = imageinfo.shape[0]
            width = imageinfo.shape[1]

            # Save new file location
            file_location = path + "/" + saved_file

            # open existing CSV file and store img height, width, and file path
            with open('cropped_image_data.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(([height, width, file_location]))
                f.close()

            # Print to terminal to confirm file is saved
            print(saved_file + " has been written to cropped_image_data.csv")


# Prompt use to select the file they wish to crop and return the file path
def upload_action(event=None):
    filename = filedialog.askopenfilename()
    return filename


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        img = cv2.imread(filename, 1)
        count = 0

        # Call main function to open window
        main()
    else:
        # Declare variables we will use in click event,
        # open image file, and set variables
        filename = upload_action()
        img = cv2.imread(filename, 1)
        count = 0

        # Call main function to open window
        main()

