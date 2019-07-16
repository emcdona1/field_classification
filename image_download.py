import csv
import pandas as pd
import requests #must do 'pip install requests' in terminal

with open('lycopodiaceae_images_univ_alaska.csv') as image_csv:
    rows = pd.read_csv(image_csv)#, delimiter = ',', quotechar='"')
    print(rows.head())
    # n_rows=list(rows)
    # image_url=n_rows[1][1]
    # coreid=n_rows[1][0]
    # print(image_url)

    # result = requests.get(image_url)
    # if (result.status_code == 200):
    #     #image = result.raw.read()
    #     with open('image_downloads/'+coreid+'.png', 'wb') as download:
    #         download.write(result.content)
    for i in range(len(rows)):
        image_url = rows.identifier[i]
        coreid = rows.coreid[i]
        result = requests.get(image_url)
        if (result.status_code == 200):
            #image = result.raw.read()
            with open('image_downloads/'+str(coreid)+'.png', 'wb') as download:
                download.write(result.content)