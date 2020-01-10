import os
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt


class ChartCreator:
    def __init__(self):
        # for roc plotting
        self.tpr = []
        self.auc = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.index = 0
        self.history = None
        self.results = pd.DataFrame()

    def update(self, history, index, tp, fn, fp, tn):
        self.history = history
        self.index = (index + 1)  # Change index from 0-based to 1-based

        num_epochs = len(history.history['loss'])
        # save the stats of the last epoch (i.e. end of the fold) to the results file
        self.results = self.results.append([[index + 1,
                                             history.history['loss'][num_epochs - 1],
                                             history.history['acc'][num_epochs - 1],
                                             history.history['val_loss'][num_epochs - 1],
                                             history.history['val_acc'][num_epochs - 1],
                                             tn, fp, fn, tp]])

    def plot_accuracy(self):
        """Create plot of training/validation accuracy, and save it to the file system.

                Parameters:
                -----
                @ history : History
                History object of results, returned from the model.fit() method.

                @ index : int
                Current fold.

                Output:
                -----
                none (file output)

                A PNG file saved in graphs folder
                """
        plt.figure(1)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Training & Validation Accuracy for Fold ' + str(self.index))
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(os.path.join('graphs', 'val_accuracy_' + str(self.index) + '.png'))
        plt.clf()

    def plot_loss(self):
        """Create plot of training/validation loss, and save it to the file system.

        Parameters:
        -----
        @ history : History
        History object of results, returned from the model.fit() method.

        @ index : int
        Current fold.

        Output:
        -----
        none (file output)

        A PNG file saved in graphs folder
        """
        plt.figure(2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Training and Validation Loss for Fold' + str(self.index))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(os.path.join('graphs', 'val_loss_' + str(self.index) + '.png'))
        plt.clf()

    def plot_roc(self, mean_auc, mean_auc_std, mean_tpr):
        """ Plot ROC and save graph to file system. """
        plt.figure(3)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
                 color='r', label='random chance', alpha=0.8)
        plt.plot(self.mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, mean_auc_std),
                 lw=2, alpha=0.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curve after ' + str(self.index) + ' folds')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('graphs', 'mean_ROC.png'))
        plt.clf()

    def generate_roc_and_auc(self, validation_labels, prediction_probability):
        """ Updates ROC plot after each fold, and saves to file system.

    	Parameters:
    	------
    	@ mean_fpr : float
    	false positive rate (mean from all folds run so far)

    	@ mean_tpr : float
    	true postive rate (mean from all folds run so far)

    	@ mean_auc : float
    	area under ROC curve (mean from all folds run so far)

    	@ std_auc : float
    	standard deviation of AUC (mean from all folds run so far)

    	Output:
    	------
    	none

    	Saves plot as `mean_ROC.png` in graphs folder.
    	"""
        # Compute values for ROC curve and area under the curve (AUC)
        latest_fpr, latest_tpr, _ = roc_curve(validation_labels, prediction_probability)
        latest_auc = auc(latest_fpr, latest_tpr)

        # Update arrays with latest ROC/AUC values
        self.tpr.append(interp(self.mean_fpr, latest_fpr, latest_tpr))
        self.tpr[-1][0] = 0.0
        self.auc.append(latest_auc)

        # find/update figures for ROC/AUC so far
        mean_tpr = np.mean(self.tpr, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        mean_auc_std = float(np.std(self.auc))

        self.plot_roc(mean_auc, mean_auc_std, mean_tpr)

    def generate_confusion_matrix(self, validation_labels, predicted_classification):
        tn, fp, fn, tp = confusion_matrix(validation_labels, predicted_classification).ravel()
        return tn, fp, fn, tp

    def save_results_to_csv(self):
        self.results.rename(columns={0: 'Fold Number', 1: 'Training Loss', 2: 'Training Accuracy',
                                     3: 'Validation Loss', 4: 'Validation Accuracy',
                                     5: 'True Negatives', 6: 'False Positives', 7: 'False Negatives',
                                     8: 'True Positives'})
        self.results.to_csv(os.path.join('graphs', 'final_acc_loss.csv'), encoding='utf-8', index=False)

    def update_and_save_graphs(self, history, index, validation_labels,
                               validation_predicted_classification, validation_predicted_probability):
        tn, fp, fn, tp = self.generate_confusion_matrix(validation_labels, validation_predicted_classification)
        self.update(history, index, tp, fn, fp, tn)
        self.plot_accuracy()
        self.plot_loss()
        self.generate_roc_and_auc(validation_labels, validation_predicted_probability)
