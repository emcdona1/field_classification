import pandas as pd
import os
import argparse
import time
import datetime
from classify_images import write_dataframe_to_CSV

def process_input_arguments():
    parser = argparse.ArgumentParser('Export predictions below a certain uncertainty.')
    parser.add_argument('-p', '--predictions', help='Filename or file path for predictions CSV')	
    parser.add_argument('-c', '--certainty', help='Threshold for certainty')
    args = parser.parse_args()

    filename = args.predictions
    threshold = float(args.certainty)

    return filename, threshold

if __name__ == '__main__':
    # Start execution and parse arguments
    start_time = time.time()
    filename, threshold = process_input_arguments()

    # Read in predictions and sort by confidence level
    predictions = pd.read_csv(filename)
    predictions = predictions.sort_values(by=['rostrata_pred'])

    # Select those tuples where BOTH predictions are below a certain threshold
    # e.g. threshold = 0.6, and predictions[1] = 0.55 and predictions[2] = 0.45
    col1 = predictions.columns[1]
    col2 = predictions.columns[2]
    below = predictions[(predictions[col1] < threshold) & (predictions[col2] < threshold)]

    # Save to file
    filepath = write_dataframe_to_CSV('predictions', 'low_confidence', below)
    print('Predictions saved to %s .' % filepath)

    # Finish execution
    end_time = time.time()
    print('Completed in %.1f seconds' % (end_time - start_time))