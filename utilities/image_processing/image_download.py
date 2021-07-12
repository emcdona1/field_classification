import csv
import pandas as pd
import requests  # must do 'pip install requests' in terminal
import argparse
import os


def download_images_from_csv(images_local_name, occurrences_local_name, image_download_location):
    with open(images_local_name, encoding='utf-8') as image_csv:
        image_rows = pd.read_csv(image_csv, usecols=['coreid', 'identifier', 'goodQualityAccessURI',
                                                     'format'])  # we only want these columns
        # delete the duplicate rows
        # reindex appropriately and drop the rows that are NaN
        image_rows = image_rows.reset_index(drop=True)
        image_rows = image_rows.drop('format', axis=1)
        print('Filtered duplicates.')
        occ_df = pd.read_csv(open(occurrences_local_name, encoding='utf-8'), usecols=['id', 'catalogNumber'])
        barcode_dict = occ_df.set_index('id').T.to_dict('list')
        # prep a way to keep track of images we couldn't find
        num_not_found = 0
        not_found = [['Barcode', 'Core ID']]
        for i in range(len(image_rows)):
            image_url = image_rows.identifier[i]
            result = requests.get(image_url)
            coreid = image_rows.coreid[i]
            barcode = barcode_dict.get(coreid)[0]
            # if the identifier link does not work, try the goodQualityAccessURI link
            if result.status_code != 200:
                image_url = image_rows.goodQualityAccessURI[i]
                result = requests.get(image_url)
            if result.status_code == 200:
                with open(image_download_location + '/' + str(barcode) + '_' + str(coreid) + '.jpg', 'wb') as download:
                    download.write(result.content)
            else:  # when both links don't work give up oh well
                num_not_found = num_not_found + 1
                not_found.append([barcode, coreid])

            # keep user updated on progress
            if i % 50 == 0:
                print('%i images downloaded.' % i)
        print('Image download complete.')
        print('%i image(s) were not found.' % num_not_found)
        if num_not_found > 0:
            with open(image_download_location + "/_not_found.csv", "w", newline="") as output_file:
                writer = csv.writer(output_file)
                writer.writerows(not_found)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-f', '--file', default='dli_images/images_final.csv',
                        help='Upload CSV with URLs in 2nd column')
    parser.add_argument('-o', '--occurrences', help="name of occurrences file to help put the barcode on")
    parser.add_argument('-l', '--location', default='',
                        help='type the location you want images downloaded in (NO SLASH AT END)')
    args = parser.parse_args()
    if not os.path.exists(args.location):
        os.makedirs(args.location)
    download_images_from_csv(args.file, args.occurrences, args.location)
