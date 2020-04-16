import pandas as pd
import sys
import argparse
import time
from datetime import datetime
import os


def process_input_arguments():
    parser = argparse.ArgumentParser('Export predictions below a certain uncertainty.')
    parser.add_argument('-p', '--predictions', help='Filename or file path for predictions CSV')
    parser.add_argument('-c', '--certainty', help='Threshold for certainty')
    parser.add_argument('-s', '--setting', default='l',
                        help='l for low confidence filter, h for high confidence filter')
    args = parser.parse_args()
    filename = args.predictions
    threshold = float(args.certainty)
    setting = args.setting

    return filename, threshold, setting


def write_dataframe_to_csv(folder, filename, dataframe_to_write):
    ''' Writes the given DataFrame to a file.
    Parameters:
    -----
    @folder : String to designate folder in which to write file
    @filename : String to add designation to filename -- file names are timestamp+filename
    @dataframe_to_write : DataFrame to be written to CSV

    Output:
    -----
    File path of the written file
    '''
    timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    filename = timestamp + filename + '.csv'
    filepath = os.path.join(folder, filename)
    dataframe_to_write.to_csv(filepath, encoding='utf-8', index=False)

    return filepath


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
    file_path = write_dataframe_to_csv('predictions', filename_option, results)
    print('Predictions saved to %s .' % file_path)

    # Finish execution
    end_time = time.time()
    print('Completed in %.1f seconds' % (end_time - start_time))
