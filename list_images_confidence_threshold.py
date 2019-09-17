import pandas as pd
import os
import sys
import argparse
import time
import datetime
from classify_images import write_dataframe_to_CSV

def process_input_arguments():
    parser = argparse.ArgumentParser('Export predictions below a certain uncertainty.')
    parser.add_argument('-p', '--predictions', help='Filename or file path for predictions CSV')	
    parser.add_argument('-c', '--certainty', help='Threshold for certainty')
    parser.add_argument('-s', '--setting', default='l', help ='l for low confidence filter, h for high confidence filter')
    args = parser.parse_args()
    filename = args.predictions
    threshold = float(args.certainty)
    setting = args.setting

    return filename, threshold, setting

if __name__ == '__main__':
    # Start execution and parse arguments
    start_time = time.time()
    filename, threshold, setting = process_input_arguments()

    # Read in predictions and sort by confidence level
    predictions = pd.read_csv(filename)
    col1 = predictions.columns[1]
    col2 = predictions.columns[2]
    predictions = predictions.sort_values(by=[col2])

    # Select those tuples where BOTH predictions are below a certain threshold
    # e.g. threshold = 0.6, and predictions[1] = 0.55 and predictions[2] = 0.45
    results = []
    filename_option = ''
    if setting == 'l':
        results = predictions[(predictions[col1] < threshold) & (predictions[col2] < threshold)]
        filename_option = 'low_confidence'
    elif setting == 'h':
        results = predictions[(predictions[col1] < (1 - threshold)) | (predictions[col2] < (1 - threshold))]
        filename_option = 'high_confidence'
    else:
        print('ERROR: Invalid setting selected. (Hint: choose l for low or h for high confidence filtering.)')
        sys.exit(0)

    # Save to file
    filepath = write_dataframe_to_CSV('predictions', filename_option, results)
    print('Predictions saved to %s .' % filepath)

    # Finish execution
    end_time = time.time()
    print('Completed in %.1f seconds' % (end_time - start_time))