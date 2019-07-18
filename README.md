# fern_classifications

# Using image_download.py
The goal of this file is to download images from the Pteridophyte Portal. Before running the file, please follow the steps below:
1. Go to [The Pteridophyte Collections Consortium](http://www.pteridoportal.org/portal/)
2. Click Search > Collections > Deselect All > Choose your source (we chose 'Field Museum of Natural History Pteridophyte Collection')
3. Hit Search in the upper right
4. Fill in your search parameters and hit 'List Display'
5. In the top right, click the little download button (it looks like a down arrow into a open box)
6. Choose the following parameters:

  *Structure: Darwin Core  
   *Data Extensions: Keep both boxes selected   
   *File Format: Comma Delimited (CSV)   
   *Character Set: UTF-8 (Unicode)   
   *Compression: Check this box   
   
7. Hit 'Download Data'

Once you have your downloaded zip file, you will want two CSV's in particular: the images and occurrences
Place these files in the folder that your code is in and create a new folder in this space where you want the images to download.

Run this code by using the following command in terminal/command prompt

`python image_download.py -f [image_csv_name].csv -o [occurrences_csv_name].csv -l [folder_name]`

For example:

`python image_download.py -f images.csv -o occurrences.csv -l specimen_images`

`python image_download.py -f lyco_images.csv -o lyco_occurrences.csv -l lyco_images`

or if the CSVs are in a folder inside the workspace:
`python image_download.py -f lyco_csvs/lyco_images.csv -o lyco_csvs/lyco_occurrences.csv -l lyco_images`

Because of efficiency purposes, the program first looks in the 'identifier' column for the image. If it's not found, it will then look in the 'goodQualityAccessURI' column. If neither produce a useable image, the program will output a CSV with the missing images in the folder you input that lists the barcodes and core id numbers.
