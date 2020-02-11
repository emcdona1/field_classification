import os
import argparse
from timer import Timer
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib
from image_handling import ImageImporter
from neural_network_models import SmithsonianModel
from data_and_visualization_io import DataChartIO

matplotlib.use('Agg')  # required when running on server

# setup
SEED = 1
np.random.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
random.seed(SEED)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001


def main() -> None:
    timer = Timer('TrainingModels')
    # Set up
    n_folds, img_directory, folders, img_size, color, n_epochs = parse_arguments()
    create_folders()

    # Load in images and shuffle order
    images = ImageImporter(img_directory, folders, img_size, color, SEED)
    features = images.features
    labels = images.labels

    # Train model
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    charts = DataChartIO()
    for index, (training_idx_list, validation_idx_list) in enumerate(skf.split(features, labels)):
        model, history, validation_features, validation_labels = model_training(n_epochs)
        model_validation()
    charts.save_results_to_csv()

    # end
    print('c1: ' + folders[0] + ', c2: ' + folders[1])
    timer.stop_timer()


def parse_arguments():
    parser = argparse.ArgumentParser('import images and train model')
    parser.add_argument('-d', '--directory', default='', help='Folder holding category folders')
    parser.add_argument('-c1', '--category1', help='Folder of class 1')
    parser.add_argument('-c2', '--category2', help='Folder of class 2')
    parser.add_argument('-s', '--img_size', default=256, help='Image dimension in pixels')
    parser.add_argument('-cm', '--color_mode', default=1, help='Color mode to use (1=color, 0=grayscale)')
    parser.add_argument('-n', '--number_folds', default=10, help='Number of folds (minimum 2) for cross validation')
    parser.add_argument('-e', '--number_epochs', default=25, help='Number of epochs')
    args = parser.parse_args()

    img_directory = args.directory
    folders = [args.category1, args.category2]
    img_size = int(args.img_size)
    n_folds = int(args.number_folds)
    n_epochs = int(args.number_epochs)
    color = True if int(args.color_mode) == 1 else False

    return n_folds, img_directory, folders, img_size, color, n_epochs


def create_folders():
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')


def model_training(num_epochs):
    # set up training/validation
    train_features = features[training_idx_list]
    train_labels = labels[training_idx_list]
    validation_features = features[validation_idx_list]
    validation_labels = labels[validation_idx_list]
    architecture = SmithsonianModel(img_size, color_mode=color, seed=SEED, lr=LEARNING_RATE)

    print('Training model for fold' + str(index + 1) + '/' + str(n_folds))
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', \
    #        mode='min', min_delta = 0.05, patience = 20, restore_best_weights = True)
    history = architecture.model.fit(train_features, train_labels,
                                     batch_size=BATCH_SIZE, epochs=num_epochs,
                                     #        callbacks = [es_callback], \
                                     validation_data=(validation_features, validation_labels),
                                     verbose=2)
    architecture.model.save(os.path.join('saved_models', 'CNN_' + str(index + 1) + '.model'))

    return architecture.model, history, validation_features, validation_labels


def model_validation():
    # Classify the validation set
    validation_predicted_probability = model.predict_proba(validation_features)[:, 1]
    validation_predicted_classification = [round(a + 0.0001) for a in validation_predicted_probability]
    charts.update_and_save_graphs(history, index, validation_labels,
                                  validation_predicted_classification, validation_predicted_probability)


if __name__ == '__main__':
    main()
