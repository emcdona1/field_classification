import os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Chart(ABC):
    def __init__(self, base_filename, folder_name):
        self.path = os.path.join(folder_name, base_filename)
        self.file_extension = '.png'

    @abstractmethod
    def update(self, index, validation_labels, prediction_probability, history, class_labels, count, cls) -> None:
        pass

    def save(self, index) -> None:
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

    def update(self, index, validation_labels, prediction_probability, history, class_labels, predictions, cls) -> None:
        # cls = 1
        # for cls in range(len(class_labels)):
        class_predictions = []

        for i in range(len(validation_labels)):
            class_predictions.append(predictions[i][cls])

            # print(predictions)
            # print(class_predictions)

            # print(validation_labels)

        for x in range(len(validation_labels)):
            if validation_labels[x] == cls:
                validation_labels[x] = 1
            elif validation_labels[x] != cls:
                validation_labels[x] = 0

                # print(validation_labels)

            print(cls)

        latest_fpr, latest_tpr, thresholds = roc_curve(validation_labels, class_predictions)
        latest_auc = roc_auc_score(validation_labels, class_predictions)

        self.fpr[index] = latest_fpr
        self.tpr[index] = latest_tpr
        self.auc[index] = latest_auc

            # cls = cls + 1
        self.create_chart(index, cls)

    def create_chart(self, index, cls) -> None:
        plt.figure(3)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve - Fold %i' % index)
        plt.title('ROC Curve - Class %i' % cls)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
        plt.plot(self.fpr[index], self.tpr[index], color='blue',
                 label='Mean ROC (AUC = %0.2f)' % (self.auc[index]),
                 lw=2, alpha=0.8)
        plt.legend(loc="lower right")

        # if cls < 4:
        #     cls = cls + 1
        #     self.create_chart(index, cls)

    def finalize(self, results) -> None:

        results['auc'] = self.auc.values()

    #     self.tpr = {}
    #     self.fpr = {}
    #     self.auc = {}
    #
    # def update(self, index, validation_labels, prediction_probability, history, class_labels) -> None:
    #     # 1. Compute ROC curve and AUC
    #
    #     # latest_fpr, latest_tpr, thresholds = roc_curve(validation_labels, prediction_probability)
    #     # latest_auc = _multiclass_roc_auc_score(validation_labels, prediction_probability, class_labels, 'ovr',
    #     #                                        'macro', sample_weight=None)
    #
    #     latest_fpr, latest_tpr, thresholds = roc_curve(validation_labels, prediction_probability)
    #     latest_auc = roc_auc_score(validation_labels, prediction_probability)
    #
    #     # 2. save new values to instance variables
    #     self.fpr[index] = latest_fpr
    #     self.tpr[index] = latest_tpr
    #     self.auc[index] = latest_auc
    #
    #     # 3. Create and save ROC chart
    #     self.create_chart(index)
    #
    # def create_chart(self, index) -> None:
    #     plt.figure(3)
    #     plt.xlim([-0.05, 1.05])
    #     plt.ylim([-0.05, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC Curve - Fold %i' % index)
    #     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
    #     plt.plot(self.fpr[index], self.tpr[index], color='blue',
    #              label='Mean ROC (AUC = %0.2f)' % (self.auc[index]),
    #              lw=2, alpha=0.8)
    #     plt.legend(loc="lower right")
    #
    # def finalize(self, results) -> None:
    #     results['auc'] = self.auc.values()


class AccuracyChart(Chart):
    def __init__(self, folder_name):
        base_filename = 'accuracy'
        super().__init__(base_filename, folder_name)

        self.training = {}
        self.validation = {}

    def update(self, index, validation_labels, prediction_probability, history, class_labels, predictions, cls) -> None:
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

    def update(self, index, validation_labels, prediction_probability, history, class_labels, predictions, cls) -> None:
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

    def update(self, index, validation_labels, prediction_probability, history, class_labels, predictions, cls) -> None:

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
        # print(len(validation_labels))
        # print(len(validation_predicted_classification))

        classes = []

        # get unique class labels and put them in order
        for x in validation_labels:
            if x not in classes:
                classes.append(x)
        classes = classes.sort()

        display_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        display_matrix.plot()

        # fig = plt.figure(4)
        # ax = fig.add_subplot(1, 1, 1)
        # ax.matshow(cm, cmap=plt.cm.get_cmap('Blues'), alpha=0.3)
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
        # cax = ax.matshow(cm)
        # fig.colorbar(cax)
        # ax.set_xticks([0, ])
        # ax.set_xticklabels(class_labels)
        # ax.set_yticks([0, 1])
        # ax.set_yticklabels(class_labels)
        # plt.title('Confusion Matrix - Fold %i' % index)
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')

    def finalize(self, results) -> None:
        correct = 0
        incorrect = 0

        for x in range(len(self.actual)):
            if self.actual[x] == self.predicted[x]:
                correct = correct + 1
            else:
                incorrect = incorrect + 1

        # print(correct)
        # print(incorrect)
        results['correct predictions'] = correct
        results['incorrect predictions'] = incorrect

        # results['tp'] = self.tp.values()
        # results['fn'] = self.fn.values()
        # results['fp'] = self.fp.values()
        # results['tn'] = self.tn.values()

    #     self.tp = {}
    #     self.fn = {}
    #     self.fp = {}
    #     self.tn = {}
    #
    # def update(self, index, validation_labels, prediction_probability, history, class_labels) -> None:
    #     # Determine the class of an image, if >= 0.4999 = class 0, otherwise class 1
    #     validation_predicted_classification = [round(a + 0.0001) for a in prediction_probability]
    #     cm = confusion_matrix(validation_labels, validation_predicted_classification)
    #
    #     # cm = confusion_matrix(validation_labels, prediction_probability, labels=class_labels)
    #     display_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    #     display_matrix.plot()
    #     plt.savefig('testCM.png')
    #
    #     new_tn, new_fp, new_fn, new_tp = cm.ravel()
    #     self.tp[index] = new_tp
    #     self.fn[index] = new_fn
    #     self.fp[index] = new_fp
    #     self.tn[index] = new_tn
    #
    #     self.create_chart(index, cm, class_labels)
    #
    # def create_chart(self, index, cm, labels) -> None:
    #
    #     fig = plt.figure(4)
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.matshow(cm, cmap=plt.cm.get_cmap('Blues'), alpha=0.3)
    #     for i in range(cm.shape[0]):
    #         for j in range(cm.shape[1]):
    #             ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
    #     cax = ax.matshow(cm)
    #     fig.colorbar(cax)
    #     ax.set_xticks([0, 1])
    #     ax.set_xticklabels(labels)
    #     ax.set_yticks([0, 1])
    #     ax.set_yticklabels(labels)
    #     plt.title('Confusion Matrix - Fold %i' % index)
    #     plt.xlabel('Predicted')
    #     plt.ylabel('Actual')
    #
    # def finalize(self, results) -> None:
    #     results['tp'] = self.tp.values()
    #     results['fn'] = self.fn.values()
    #     results['fp'] = self.fp.values()
    #     results['tn'] = self.tn.values()
