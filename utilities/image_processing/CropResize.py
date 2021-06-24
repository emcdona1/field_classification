import cv2
import numpy as np
import csv


def main():
    resize()


def find_new_dimensions():
    height = 0
    width = 0
    max_height = 0
    max_width = 0
    with open('CroppedImages.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row != []:
                # print(row)
                height = int(row[0])
                width = int(row[1])
                if height > max_height:
                    max_height = height
                if width > max_width:
                    max_width = width
    print(max_height, max_width)
    return max_height, max_width


def resize():
    height, width = find_new_dimensions()
    count = 0

    with open('CroppedImages.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row != []:
                path = str(row[2])
                # path = os.path.normpath(path)
                # read image
                img = cv2.imread(path)
                print(path)

                ht, wd, cc = img.shape

                # create new image of desired size and color (blue) for padding
                ww = width
                hh = height
                color = (252, 248, 245)
                result = np.full((hh, ww, cc), color, dtype=np.uint8)

                # compute center offset
                xx = (ww - wd) // 2
                yy = (hh - ht) // 2

                # copy img image into center of result image
                result[yy:yy + ht, xx:xx + wd] = img

                # # view result
                # cv2.imshow("result", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                count += 1

                # Split our filename to get the desired sections
                # and exclude the .jpg or .png extension
                path_split = path.split("/")
                species = path_split[-2]
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
                print('saved')


if __name__ == "__main__":
    main()
