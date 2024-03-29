import math
import os
from models import CNNModel
from labeled_images.labeledimages import LabeledImages
import numpy as np
from data_visualization.visualizationgenerator import VisualizationGenerator
from tensorflow import keras
import tensorflow as tf


class ModelTrainer:
    def __init__(self, epochs: int, batch_size: int, architecture: CNNModel, seed: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.folder_name = 'saved_models'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.architecture = architecture
        self.curr_index = 1
        self.history = list()
        self.seed: int = seed
        self.charts = VisualizationGenerator()
        self.class_weight = {}

    def train_and_save_all_models(self, images: LabeledImages):
        self.train_model(images)
        keras.models.save_model(self.architecture.model,
                                os.path.join(self.folder_name, f'CNN_1.model'))
        self.validate_model_at_epoch_end(images, images.validation_image_set)

    def train_model(self, images: LabeledImages) -> None:
        self.class_weight = dict.fromkeys(range(0, len(images.class_labels)))
        for curr_ind in images.count_per_class:
            curr_count = float(images.count_per_class[curr_ind])
            print(f'Items in class {curr_ind}: {(int)(curr_count)}')
            weight = images.img_count / (10 * math.log(curr_count, 2))                  # total/[10*log2(class)]
            # weight = images.img_count / (4 * curr_count)                                # total/[4*class]
            # weight = images.img_count / (2 * curr_count)                                # total/[2*class]
            # weight = (10 * images.img_count) / (math.log(curr_count, 2) * curr_count)   # [total * 10]/[(class)*log2(class)]
            # weight = (images.img_count/4.0 - curr_count) / (math.log(curr_count, 2))    # [total/4 - class]/[log2(class)]
            self.class_weight[curr_ind] = weight
        # self.class_weight = {0: 1, 1: 1}      # manual weights
        print(f'Weights: {self.class_weight}')
        self.architecture.reset_model()
        print(f'Training model .')
        new_history = self.architecture.model.fit(images.training_image_set,
                                                  validation_data=images.validation_image_set,
                                                  batch_size=self.batch_size,
                                                  epochs=self.epochs,
                                                  verbose=2,
                                                  class_weight=self.class_weight        # comment this line if no weights
                                                  )
        self.history.append(new_history)

    def validate_model_at_epoch_end(self, images: LabeledImages, validation_set: tf.data.Dataset) -> None:
        validation_labels = np.array([])
        validation_predicted_probability = self.architecture.model.predict(validation_set)[:, 1]  # ,0]
        for batch in validation_set.as_numpy_iterator():
            validation_labels = np.concatenate((validation_labels, batch[1]))
        validation_labels = validation_labels.astype(np.int32)
        predictions = self.architecture.model.predict(validation_set)
        current_class = 0

        print('Classes: ', len(images.class_labels))

        self.charts.update(validation_labels, validation_predicted_probability,
                           self.history[0], images.class_labels, predictions, current_class)
