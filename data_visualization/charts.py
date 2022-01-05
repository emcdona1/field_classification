import os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Chart(ABC):
    def __init__(self, base_filename, folder_name):
        self.path = os.path.join(folder_name, base_filename)
        self.file_extension = '.png'

    @abstractmethod
    def update(self, index, validation_labels, prediction_probability, history, class_labels, count,
               current_cls) -> None:
        pass

    def save(self, index, class_labels, current_cls) -> None:
        plt.savefig(self.path + str(index).zfill(2) + self.file_extension)
        plt.clf()

    @abstractmethod
    def finalize(self, results) -> None:
        pass


class ROCChart(Chart):

    def __init__(self, folder_name):
        base_filename = 'mean_ROC'
        super().__init__(base_filename, folder_name)

        self.tpr = {}
        self.fpr = {}
        self.auc = {}

    def update(self, index, validation_labels, prediction_probability, history, class_labels, predictions,
               current_cls) -> None:

        for cls in range(len(class_labels)):
            current_cls = cls
            class_predictions = []

            for i in range(len(validation_labels)):
                class_predictions.append(predictions[i][current_cls])

            labels = validation_labels.copy()

            for x in range(len(validation_labels)):
                if validation_labels[x] == current_cls:
                    labels[x] = 1
                else:
                    labels[x] = 0

            # Compute ROC curve and AUC
            latest_fpr, latest_tpr, thresholds = roc_curve(labels, class_predictions)
            latest_auc = roc_auc_score(labels, class_predictions)

            # save new values to instance variables
            self.fpr[index] = latest_fpr
            self.tpr[index] = latest_tpr
            self.auc[index] = latest_auc

            # create ROC chart
            self.create_chart(index, current_cls, class_labels)

    # override save method to loop through classes and fave files seperatly
    def save(self, index, class_labels, current_cls) -> None:
        # if len(class_labels) > 2:
        #     for c in range(len(class_labels)):
        #         print(self.path + '_Class' + str(c).zfill(2) + self.file_extension)
        #         plt.savefig(self.path + '_Class' + str(c).zfill(2) + self.file_extension)
        #         plt.clf()
        # else:
        #     print(self.path + str(current_cls).zfill(2) + self.file_extension)
        #     plt.savefig(self.path + str(current_cls).zfill(2) + self.file_extension)
        #     plt.clf()

        # class_labels[current_cls] = str(current_cls)
        # for c in class_labels:
        #     if int(class_labels[c]) == current_cls:
        #         print(class_labels[c])
        #         pass

        if current_cls < len(class_labels):
            if os.path.exists(self.path + '_Class' + str(current_cls).zfill(2) + self.file_extension):
                pass
            elif len(class_labels) == 2:
                plt.savefig(self.path + 'Binary' + str(index).zfill(2) + self.file_extension)
            else:
                print('save current cls: ', current_cls)
                plt.savefig(self.path + '_Class' + str(current_cls).zfill(2) + self.file_extension)
                plt.clf()

    def create_chart(self, index, cls, class_labels) -> None:
        plt.figure(1 + cls)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if len(class_labels) == 2:
            plt.title('ROC Curve - Fold %i' % index)
        else:
            plt.title('ROC Curve - Class %i' % cls)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
        plt.plot(self.fpr[index], self.tpr[index], color='blue',
                 label='Mean ROC (AUC = %0.2f)' % (self.auc[index]),
                 lw=2, alpha=0.8)
        plt.legend(loc="lower right")
        print("create class: ", cls)

        self.save(index, class_labels, cls)

    def finalize(self, results) -> None:
        results['auc'] = self.auc.values()


class AccuracyChart(Chart):
    def __init__(self, folder_name):
        base_filename = 'accuracy'
        super().__init__(base_filename, folder_name)

        self.training = {}
        self.validation = {}

    def update(self, index, validation_labels, prediction_probability, history, class_labels, predictions,
               current_cls) -> None:
        """Create plot of training/validation accuracy, and save it to the file system."""
        self.training[index] = history.history['accuracy'][-1]
        self.validation[index] = history.history['val_accuracy'][-1]
        self.create_chart(index, history)

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

    def update(self, index, validation_labels, prediction_probability, history, class_labels, predictions,
               current_cls) -> None:
        self.training[index] = history.history['loss'][-1]
        self.validation[index] = history.history['val_loss'][-1]
        self.create_chart(index, history)

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
        base_filename = 'confusion_matrix'
        super().__init__(base_filename, folder_name)

        self.predicted = {}
        self.actual = {}

    def update(self, index, validation_labels, prediction_probability, history, class_labels, predictions,
               current_cls) -> None:

        validation_predicted_classification = []
        cls = 0
        for i in range(len(validation_labels)):
            max_val = 0
            for j in range(len(class_labels)):
                if predictions[i][j] > max_val:
                    max_val = predictions[i][j]
                    cls = j
            predicted = cls
            validation_predicted_classification.append(predicted)

        cm = confusion_matrix(validation_labels, validation_predicted_classification)

        self.actual = validation_labels
        self.predicted = validation_predicted_classification
        print(class_labels)

        classes = []

        # get unique class labels and put them in order
        for x in validation_labels:
            if x not in classes:
                classes.append(x)
        classes = classes.sort()

        display_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        display_matrix.plot()

    def finalize(self, results) -> None:
        correct = 0
        incorrect = 0

        for x in range(len(self.actual)):
            if self.actual[x] == self.predicted[x]:
                correct = correct + 1
            else:
                incorrect = incorrect + 1

        results['correct predictions'] = correct
        results['incorrect predictions'] = incorrect
