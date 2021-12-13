import os
import pandas as pd
from models import CNNModel
from labeled_images.labeledimages import LabeledImages, MAX_LABEL_LENGTH, num_to_char
import numpy as np
from data_visualization.visualizationgenerator import VisualizationGenerator
from tensorflow import keras
import tensorflow as tf


class ModelTrainer:
    def __init__(self, epochs: int, batch_size: int, n_folds: int, architecture: CNNModel, seed: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.folder_name = 'saved_models'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.architecture = architecture
        self.n_folds = n_folds
        self.curr_index = 1
        self.history = list()
        self.seed: int = seed
        self.charts = VisualizationGenerator(self.n_folds)

    def train_and_save_all_models(self, images: LabeledImages):
        fold_groups = zip(images.training_image_set, images.validation_image_set)
        for index, (training_set, validation_set) in enumerate(fold_groups):
            self.curr_index = index
            self.train_model(training_set, validation_set)
            keras.models.save_model(self.architecture.model,
                                    os.path.join(self.folder_name, f'CNN_{self.curr_index + 1}.model'))
            self.validate_model_at_epoch_end(images, validation_set)

    def train_model(self, training_set: tf.data.Dataset, validation_set: tf.data.Dataset) -> None:
        self.architecture.reset_model()
        print(f'Training model for fold {self.curr_index + 1} of {self.n_folds}.')
        new_history = self.architecture.model.fit(training_set,
                                                  validation_data=validation_set,
                                                  batch_size=self.batch_size,
                                                  epochs=self.epochs,
                                                  verbose=2)
        self.history.append(new_history)

    def validate_model_at_epoch_end(self, images: LabeledImages, validation_set: tf.data.Dataset) -> None:
        validation_labels = np.array([])
        validation_predicted_probability = self.architecture.model.predict(validation_set)[:, 1]  # ,0]
        for batch in validation_set.as_numpy_iterator():
            validation_labels = np.concatenate((validation_labels, batch[1]))
        validation_labels = validation_labels.astype(np.int32)

        print(len(images.class_labels))
        # if len(images.class_labels) == 2:
        self.charts.update(self.history[self.curr_index], self.curr_index + 1, validation_labels,
                            validation_predicted_probability, images.class_labels)


def decode_batch_predictions(predictions: np.ndarray):
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(predictions, input_length=input_len, greedy=True)[0][0][:, :MAX_LABEL_LENGTH]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        res = res.replace('[UNK]', '').replace(' ', '')
        output_text.append(res)
    return output_text


class CtcModelTrainer(ModelTrainer):
    def validate_model_at_epoch_end(self, images: LabeledImages, validation_set: tf.data.Dataset) -> None:
        predictions = pd.DataFrame(columns=['actual', 'predicted'])
        for batch in validation_set.as_numpy_iterator():
            validation_predicted_probability = self.architecture.model.predict(batch)
            predicted_labels = decode_batch_predictions(validation_predicted_probability)
            actual_labels_as_byte_lists = num_to_char(batch['label']).numpy()
            actual_labels = list()
            for row in actual_labels_as_byte_lists:
                row = [c.decode('utf-8') for c in row]
                row = ''.join(row).strip()
                actual_labels.append(row)
            batch_predictions = pd.DataFrame({'actual': actual_labels, 'predicted': predicted_labels})
            predictions = predictions.append(batch_predictions, ignore_index=True)
        print(predictions)
        # todo: output these results somehow