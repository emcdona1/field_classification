import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from abc import abstractmethod


class Charts:
    def __init__(self):
        self.all_charts = []
        self.all_charts.append(ROCChart())
        self.all_charts.append(AccuracyChart())
        self.all_charts.append(LossChart())
        self.all_charts.append(ConfusionMatrix())

    def update(self, history, index, validation_labels, prediction_probability, args):
        for each in self.all_charts:
            each.update(index, validation_labels, prediction_probability, history, args)

    def finalize(self):
        for each in self.all_charts:
            each.finalize()


class Chart:
    def __init__(self, base_filename):
        self.path = os.path.join('graphs', base_filename)
        self.file_extension = '.png'

    @abstractmethod
    def update(self, index, validation_labels, prediction_probability, history, args):
        pass

    def save(self, index):
        plt.savefig(self.path + str(index) + self.file_extension)
        plt.clf()

    @abstractmethod
    def finalize(self):
        pass


class ROCChart(Chart):
    def __init__(self):
        base_filename = 'mean_ROC'
        super().__init__(base_filename)

        self.tpr = {}
        self.fpr = {}
        self.auc = {}

    def update(self, index, validation_labels, prediction_probability, history, args):
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
        # 1. Compute ROC curve and AUC
        latest_fpr, latest_tpr, thresholds = roc_curve(validation_labels, prediction_probability)
        latest_auc = roc_auc_score(validation_labels, prediction_probability)

        # 2. save new values to instance variables
        self.fpr[index] = latest_fpr
        self.tpr[index] = latest_tpr
        self.auc[index] = latest_auc

        # 3. Create and save ROC chart
        self.create_chart(index)
        self.save(index)

    def create_chart(self, index):
        plt.figure(3)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Fold %i' % index)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
        plt.plot(self.fpr[index], self.tpr[index], color='blue',
                 label='Mean ROC (AUC = %0.2f)' % (self.auc[index]),
                 lw=2, alpha=0.8)
        plt.legend(loc="lower right")

    def finalize(self):
        # TODO
        pass


class AccuracyChart(Chart):
    def __init__(self):
        base_filename = 'accuracy'
        super().__init__(base_filename)

        self.training = {}
        self.validation = {}

    def update(self, index, validation_labels, prediction_probability, history, args):
        """Create plot of training/validation accuracy, and save it to the file system."""
        self.training[index] = history.history['acc']
        self.validation[index] = history.history['val_acc']

        self.create_chart(index)
        self.save(index)

    def create_chart(self, index):
        plt.figure(1)
        plt.plot(self.training[index], label='Training Accuracy')
        plt.plot(self.validation[index], label='Validation Accuracy')
        plt.title('Accuracy - Fold %i' % index)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

    def finalize(self):
        # TODO
        pass


class LossChart(Chart):
    def __init__(self):
        base_filename = 'loss'
        super().__init__(base_filename)

        self.training = {}
        self.validation = {}

    def update(self, index, validation_labels, prediction_probability, history, args):
        self.training[index] = history.history['loss']
        self.validation[index] = history.history['val_loss']

        self.create_chart(index)
        self.save(index)

    def create_chart(self, index):
        plt.figure(2)
        plt.plot(self.training[index], label='Training Loss')
        plt.plot(self.validation[index], label='Validation Loss')
        plt.title('Loss - Fold %i' % index)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

    def finalize(self):
        # todo
        pass


class ConfusionMatrix(Chart):
    def __init__(self):
        base_filename = 'confusion_matrix'
        super().__init__(base_filename)

        self.tp = {}
        self.fn = {}
        self.fp = {}
        self.tn = {}

    def update(self, index, validation_labels, prediction_probability, history, args):
        # Determine the class of an image, if >= 0.4999 = class 0, otherwise class 1
        validation_predicted_classification = [round(a + 0.0001) for a in prediction_probability]
        cm = confusion_matrix(validation_labels, validation_predicted_classification)

        new_tn, new_fp, new_fn, new_tp = cm.ravel()
        self.tp[index] = new_tp
        self.fn[index] = new_fn
        self.fp[index] = new_fp
        self.tn[index] = new_tn

        labels = [args.c1, args.c2]
        self.create_chart(index, cm, labels)
        self.save(index)

    def create_chart(self, index, cm, labels):
        fig = plt.figure(4)
        ax = fig.add_subplot(1, 1, 1)  # todo: figure out what this is about
        cax = ax.matshow(cm)  # todo: figure out what this is about
        plt.title('Confusion Matrix - Fold %i' % index)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    def finalize(self):
        # todo
        pass


class DataChartIO:
    def __init__(self):
        self.index = 0
        self.history = None
        self.results = pd.DataFrame()

    def update_values(self, history, index):  # todo: convert this to the finalize() methods
        self.history = history
        self.index = (index + 1)  # Change index from 0-based to 1-based

        num_epochs = len(history.history['loss'])
        # save the stats of the last epoch (i.e. end of the fold) to the results file
        self.results = self.results.append([[index + 1,
                                             history.history['loss'][num_epochs - 1],
                                             history.history['acc'][num_epochs - 1],
                                             history.history['val_loss'][num_epochs - 1],
                                             history.history['val_acc'][num_epochs - 1]]])

    def save_results_to_csv(self):
        self.results.rename(columns={0: 'Fold Number', 1: 'Training Loss', 2: 'Training Accuracy',
                                     3: 'Validation Loss', 4: 'Validation Accuracy', })
        self.results.to_csv(os.path.join('graphs', 'final_acc_loss.csv'), encoding='utf-8', index=False)

    def update_and_save_graphs(self, history, index, validation_labels,
                               validation_predicted_classification, validation_predicted_probability, args):
        self.update_values(history, index)
