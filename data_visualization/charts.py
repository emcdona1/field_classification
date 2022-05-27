import os
import numpy as np
from typing import Union
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History


class Chart(ABC):
    def __init__(self, base_filename, folder_name: Union[str, Path]):
        self.path = os.path.join(folder_name, base_filename)
        self.file_extension = '.png'

    @abstractmethod
    def update(self, current_fold_index: int,
               validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               count: np.array,
               current_class: int) -> None:
        pass

    def save(self, current_fold_index: int, class_labels, current_class) -> None:
        plt.savefig(self.path + str(current_fold_index).zfill(2) + self.file_extension)
        plt.clf()

    @abstractmethod
    def finalize(self, results) -> None:
        pass


class ROCChart(Chart):

    def __init__(self, folder_name):
        base_filename = 'mean_ROC'
        super().__init__(base_filename, folder_name)

        # initialize instance variables
        self.tpr = {}
        self.fpr = {}
        self.auc = {}

    def update(self, current_fold_index: int,
               validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               count: np.array,
               current_class: int) -> None:
        # assign current_class value
        for cls in range(len(class_labels)):
            current_class = cls
            class_predictions = []

            # fill class_predictions with the correct class predictions
            for i in range(len(validation_labels)):
                class_predictions.append(predictions[i][current_class])

            # copy image labels to be binarized
            labels = validation_labels.copy()

            # binarized labels and confirm at least one correct prediction
            valid = False
            for x in range(len(validation_labels)):
                if validation_labels[x] == current_class:
                    labels[x] = 1
                    valid = True
                else:
                    labels[x] = 0

            # if no correct predictions are present, error message is printed
            if not valid:
                print("Class ", current_class, "has no correct values, thus no ROC was generated")

            # ROC calculation and chart creation only occur when the predictions are valid
            # (has at least one positive value) and will not cause an error
            if valid:
                # Compute ROC curve and AUC
                latest_fpr, latest_tpr, thresholds = roc_curve(labels, class_predictions)
                latest_auc = roc_auc_score(labels, class_predictions)

            # save new values to instance variables
                self.fpr[current_fold_index] = latest_fpr
                self.tpr[current_fold_index] = latest_tpr
                self.auc[current_fold_index] = latest_auc

            # create ROC chart
                self.create_chart(current_fold_index, current_class, class_labels)

    # override save method to save file if that file does not already exist (needed to handle runtime error)
    def save(self, current_fold_index, class_labels, current_class) -> None:
        # run if the class value is valid
        if current_class < len(class_labels):
            # if the file exists already, pass
            if os.path.exists(self.path + '_Class' + str(current_class).zfill(2) + self.file_extension):
                pass
            # if binary, generate one chart
            elif len(class_labels) == 2:
                plt.savefig(self.path + 'Binary' + str(current_fold_index).zfill(2) + self.file_extension)
            # if multiclass, generate one graph for each class
            else:
                plt.savefig(self.path + '_Class' + str(current_class).zfill(2) + self.file_extension)
                plt.clf()

    def create_chart(self, index, cls, class_labels) -> None:
        plt.figure(3 + cls)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # label corresponding to binary or multiclass
        if len(class_labels) == 2:
            plt.title('ROC Curve - Fold %i' % index)
        else:
            plt.title('ROC Curve - Class %i' % cls)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
        plt.plot(self.fpr[index], self.tpr[index], color='blue',
                 label='Mean ROC (AUC = %0.2f)' % (self.auc[index]),
                 lw=2, alpha=0.8)
        plt.legend(loc="lower right")

        # save chart
        self.save(index, class_labels, cls)

    def finalize(self, results) -> None:
        results['auc'] = self.auc.values()


class AccuracyChart(Chart):
    def __init__(self, folder_name):
        base_filename = 'accuracy'
        super().__init__(base_filename, folder_name)

        self.training = {}
        self.validation = {}

    def update(self, current_fold_index: int,
               validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               count: np.array,
               current_class: int) -> None:
        """Create plot of training/validation accuracy, and save it to the file system."""
        self.training[current_fold_index] = history.history['accuracy'][-1]
        self.validation[current_fold_index] = history.history['val_accuracy'][-1]
        self.create_chart(current_fold_index, history)

    def create_chart(self, index, history) -> None:
        plt.figure(1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy - Fold %i' % index)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

    def finalize(self, results) -> None:
        results['training_acc'] = self.training.values()
        results['validation_acc'] = self.validation.values()


class LossChart(Chart):
    def __init__(self, folder_name):
        base_filename = 'loss'
        super().__init__(base_filename, folder_name)

        self.training = {}
        self.validation = {}

    def update(self, current_fold_index: int,
               validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               count: np.array,
               current_class: int) -> None:
        self.training[current_fold_index] = history.history['loss'][-1]
        self.validation[current_fold_index] = history.history['val_loss'][-1]
        self.create_chart(current_fold_index, history)

    def create_chart(self, index, history) -> None:
        plt.figure(2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss - Fold %i' % index)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

    def finalize(self, results) -> None:
        results['training_loss'] = self.training.values()
        results['validation_loss'] = self.validation.values()


class ConfusionMatrix(Chart):
    def __init__(self, folder_name):
        base_filename = 'validation_confusion_matrix'
        super().__init__(base_filename, folder_name)

        # initialize instance variables
        self.predicted = {}
        self.actual = {}

    def update(self, current_fold_index: int,
               validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               predictions: np.array,
               current_class: int) -> None:
        # initialize prediction list and class value
        validation_predicted_classification = []
        cls = 0

        # find the highest prediction value and determine which class
        # it represents, then store that predicted class
        for i in range(len(validation_labels)):
            max_val = 0
            for j in range(len(class_labels)):
                if predictions[i][j] > max_val:
                    max_val = predictions[i][j]
                    cls = j
            predicted = cls
            validation_predicted_classification.append(predicted)

        # generate confusion matrix for validation set
        cm = confusion_matrix(validation_labels, validation_predicted_classification)

        # store labels and predictions
        self.actual = validation_labels
        self.predicted = validation_predicted_classification

        # get unique class labels and put them in order
        classes = []
        for x in validation_labels:
            if x not in classes:
                classes.append(x)
        classes = classes.sort()

        # display confusion matrix
        display_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        display_matrix.plot()

    def finalize(self, results) -> None:
        correct = 0
        incorrect = 0

        # compare the labels to the predictions and find the
        # number of correct predictions and incorrect predictions
        for x in range(len(self.actual)):
            if self.actual[x] == self.predicted[x]:
                correct = correct + 1
            else:
                incorrect = incorrect + 1

        # store correct predictions and incorrect predictions in results
        results['correct predictions'] = correct
        results['incorrect predictions'] = incorrect
