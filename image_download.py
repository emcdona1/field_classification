import csv
import pandas as pd
import requests #must do 'pip install requests' in terminal
import argparse

def download_images_from_csv(csv_local_name, download_location):
    with open(csv_local_name) as image_csv:
        rows = pd.read_csv(image_csv, usecols=['coreid','identifier'])
        # print(rows.head())
        for i in range(len(rows)):
            image_url = rows.identifier[i]
            coreid = rows.coreid[i]
            result = requests.get(image_url)
            if (result.status_code == 200):
                with open(download_location+str(coreid)+'.png', 'wb') as download:
                    download.write(result.content)

if __name__== '__main__':
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-f', '--file', default= 'lycopodiaceae_images_univ_alaska_short.csv', help='Upload CSV with URLs in 2nd column')
    parser.add_argument('-l', '--location', default= '', help = 'type the location you want images downloaded in')
    args = parser.parse_args()
    download_images_from_csv(args.file, args.location)

#calling syntax
# python image_download.py -f [file_name] -l [location]
