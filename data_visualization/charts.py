import os
import numpy as np
from typing import Union
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History


class Chart(ABC):
    def __init__(self, folder_name: Union[str, Path], base_filename: str):
        self.path: Path = Path(folder_name, base_filename)
        self.file_extension: str = '.png'

    @abstractmethod
    def update(self,
               validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               count: np.array,
               current_class: int) -> None:
        pass

    def save(self, class_labels: list, current_class: int) -> None:
        plt.savefig(str(self.path) + self.file_extension)
        plt.clf()

    @abstractmethod
    def finalize(self, results: pd.DataFrame) -> None:
        pass


class ROCChart(Chart):
    def __init__(self, folder_name):
        base_filename = 'mean_ROC'
        super().__init__(folder_name, base_filename)
        self.tpr = {}
        self.fpr = {}
        self.auc = {}

    def update(self, validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               predictions: np.array,
               current_class: int) -> None:

        for current_class in range(len(class_labels)):
            class_predictions = []
            # fill class_predictions with the correct class predictions
            for i in range(len(validation_labels)):
                class_predictions.append(predictions[i][current_class])
            labels = [1 if curr == current_class else 0 for curr in validation_labels]

            valid = np.any(validation_labels == current_class)
            if valid:
                latest_fpr, latest_tpr, _ = roc_curve(labels, class_predictions)
                latest_auc = roc_auc_score(labels, class_predictions)
                self.fpr[1] = latest_fpr
                self.tpr[1] = latest_tpr
                self.auc[1] = latest_auc

                self.create_chart(class_labels, current_class)
            else:
                print(f'Class {current_class} has no correct values -- ROC cannot be generated.')
                self.fpr[1] = -1
                self.tpr[1] = -1
                self.auc[1] = -1

    def save(self, class_labels: list, current_class: int) -> None:
        # if in bounds
        if current_class < len(class_labels):
            # if the file does not exist already
            if not os.path.exists(f'{self.path}-Class_{current_class}{self.file_extension}'):
                if len(class_labels) == 2:
                    plt.savefig(f'{self.path}-Binary{self.file_extension}')
                else:
                    plt.savefig(f'{self.path}-Class_{current_class}{self.file_extension}')
                    plt.clf()

    def create_chart(self, class_labels: list, current_class_number: int) -> None:
        plt.figure(3 + current_class_number)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # label corresponding to binary or multiclass
        if len(class_labels) == 2:
            plt.title(f'ROC Curve')
        else:
            plt.title(f'ROC Curve - Class {class_labels[current_class_number]}')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
        plt.plot(self.fpr[1], self.tpr[1], color='blue',
                 label=f'Mean ROC (AUC = {self.auc[1]:0.2f}',
                 lw=2, alpha=0.8)
        plt.legend(loc='lower right')

        self.save(class_labels, current_class_number)

    def finalize(self, results: pd.DataFrame) -> None:
        results['auc'] = self.auc.values()


class AccuracyChart(Chart):
    def __init__(self, folder_name):
        base_filename = 'accuracy'
        super().__init__(folder_name, base_filename)
        self.training = {}
        self.validation = {}

    def update(self,
               validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               count: np.array,
               current_class: int) -> None:
        """Create plot of training/validation accuracy, and save it to the file system."""
        self.training[1] = history.history['accuracy'][-1]
        self.validation[1] = history.history['val_accuracy'][-1]
        self.create_chart(history)

    def create_chart(self, history: History) -> None:
        plt.figure(1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

    def finalize(self, results: pd.DataFrame) -> None:
        results['training_acc'] = self.training.values()
        results['validation_acc'] = self.validation.values()


class LossChart(Chart):
    def __init__(self, folder_name):
        base_filename = 'loss'
        super().__init__(folder_name, base_filename)

        self.training = {}
        self.validation = {}

    def update(self, validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               count: np.array,
               current_class: int) -> None:
        self.training[1] = history.history['loss'][-1]
        self.validation[1] = history.history['val_loss'][-1]
        self.create_chart(history)

    def create_chart(self, history) -> None:
        plt.figure(2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

    def finalize(self, results: pd.DataFrame) -> None:
        results['training_loss'] = self.training.values()
        results['validation_loss'] = self.validation.values()


class ConfusionMatrix(Chart):
    def __init__(self, folder_name):
        base_filename = 'validation_confusion_matrix'
        super().__init__(folder_name, base_filename)
        self.predicted_labels = {}
        self.actual_labels = {}
        self.confusion_matrix = np.ndarray([0])

    def update(self,
               validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               predictions: np.array,
               current_class: int) -> None:
        self.actual_labels = validation_labels
        # self.predicted_labels = np.argmax(predictions, axis=1)  # todo: verify the axis, then scrap lines below
        self.predicted_labels = []
        cls = 0
        # find the highest prediction value and determine which class
        # it represents, then store that predicted class
        for i in range(len(self.actual_labels)):
            max_val = 0
            for j in range(len(class_labels)):
                if predictions[i][j] > max_val:
                    max_val = predictions[i][j]
                    cls = j
            predicted = cls
            self.predicted_labels.append(predicted)

        self.confusion_matrix = confusion_matrix(self.actual_labels, self.predicted_labels)
        classes = list(set(self.actual_labels)).sort()
        ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=classes).plot()

    def finalize(self, results: pd.DataFrame) -> None:
        results['correct predictions'] = self.confusion_matrix.diagonal().sum()
        results['incorrect predictions'] = self.confusion_matrix.sum() - results['correct predictions']
