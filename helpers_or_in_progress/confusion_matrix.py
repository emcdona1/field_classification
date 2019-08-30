from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns
import numpy as np
import os

def build_confusion_matrix(predict_csv, dest_path, categories):
    predict = pd.read_csv(open(predict_csv, encoding='utf-8')) #predict is now a dataframe
    true_labels = list(predict.iloc[:,1])
    predicted_labels = list(predict.iloc[:,2])
    cm = confusion_matrix(true_labels, predicted_labels, labels = ['adiantum_holdoff_200', 'blechnum_holdoff_200'])
    print(cm)
    df_cm = pd.DataFrame(cm, index = categories, columns = categories)
    plt.figure()
    sns.heatmap(df_cm, annot = True, annot_kws = {"ha":"left","va":"center"}, cmap="YlGnBu", fmt="d", cbar=False, square = True)
    plt.savefig(os.path.join(dest_path,'confusion_matrix.png'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('import prediction file with true label in second column and predicted labels in third column')
    parser.add_argument('-c', '--csv_path', default='ab_results.csv', help='path to the csv file')
    parser.add_argument('-d', '--destination', default ='', help='folder where you want graph to be exported')
    args = parser.parse_args()
    categories=['adiantum','blechnum']
    build_confusion_matrix(args.csv_path, args.destination, categories)