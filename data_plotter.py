import os
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt


class ChartCreator:
    def __init__(self):
        # for roc plotting
        self.tprs = []
        self.aucs = []
        self.mean_fpr = np.linspace(0, 1, 100)

    def plot_accuracy(self, history, index):
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
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Training & Validation Accuracy for Fold ' + str(index + 1))
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(os.path.join('graphs', 'val_accuracy_' + str(index + 1) + '.png'))
        plt.clf()

    def plot_loss(self, history, index):
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
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Training and Validation Loss for Fold' + str(index + 1))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(os.path.join('graphs', 'val_loss_' + str(index + 1) + '.png'))
        plt.clf()

    def plot_roc_curve(self, index, validation_labels, prediction_probability):
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
        fpr, tpr, _ = roc_curve(validation_labels, prediction_probability)
        roc_auc = auc(fpr, tpr)

        # Update arrays with latest ROC/AUC values
        self.tprs.append(interp(self.mean_fpr, fpr, tpr))
        self.tprs[-1][0] = 0.0
        self.aucs.append(roc_auc)

        # find/update figures for ROC/AUC so far
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        mean_auc_std = float(np.std(self.aucs))

        # plot ROC and save to file system
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
        plt.title('Receiver operating characteristic (ROC) curve after ' + str(index + 1) + ' folds')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('graphs', 'mean_ROC.png'))
        plt.clf()

    def generate_confusion_matrix(self, validation_labels, predicted_classification):
        tn, fp, fn, tp = confusion_matrix(validation_labels, predicted_classification).ravel()
        return tn, fp, fn, tp
