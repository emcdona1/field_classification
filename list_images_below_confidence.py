import pandas as pd
import os
import datetime

filename = input('File name for predictions: ')
predictions = pd.read_csv('predictions\\' + filename)

# sort by confidence level
predictions = predictions.sort_values(by=['coastal_pred'])

# get confidence threshold & filter 
threshold = float(input('Desired confidence level [0,1]: '))
below = predictions[predictions['coastal_pred'] < threshold]

# save to file
timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
below.to_csv(os.path.join('predictions',timestamp+'predictions-' + str(threshold) + 'conf.csv'), encoding='utf-8',index=False)