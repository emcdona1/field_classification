# Convolutional Neural Networks and Microplants
### _utilities/zooniverse_reproductive_structures_ directory

##### About:

These scripts, run in the order below, process the raw Zooniverse classification 
file from the [Unfolding of Microplant Mysteries project](https://www.zooniverse.org/projects/nvuitton/unfolding-of-microplant-mysteries)
and turns them into a summary manifest (`prepare_csv_data.py`) and then sorts
the images into folders (`sort_images_with_voted.py`).  These images are then ready 
to use in neural network
training (e.g. `train_models_image_classification.py` in the root project folder).

##### Files:
- **prepare_csv_data.py**
  - Creates cleaned and voted CSVs from given Zooniverse Classification CSV
- **sort_images_with_voted.py**
  - Sorts given images into categories based on voted csv
