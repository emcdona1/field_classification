import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import interp
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib
from image_importer import ImageLoader
from smithsonian_model import SmithsonianModel
from data_plotter import ChartCreator

matplotlib.use('Agg')  # required when running on server

# setup
SEED = 1
np.random.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
random.seed(SEED)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001


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


def plot_accuracy_and_loss(history, index):
    """Create plots of accuracy and loss, save to disk.

    Parameters:
    -----
    @ history : History
    History object of results, returned from the model.fit() method.

    @ index : int
    Current fold.

    Output:
    -----
    none (file output)

    Two PNG files saved in graphs folder
    """

    # Save a graph of the testing/training accuracy during the training phase
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join('graphs', 'val_accuracy_' + str(index + 1) + '.png'))
    plt.clf()

    # Save a graph of the testing/training loss during the training phase
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join('graphs', 'val_loss_' + str(index + 1) + '.png'))
    plt.clf()


def plot_roc_for_kfold(mean_fpr, mean_tpr, mean_auc, std_auc):
    """ Update and save mean ROC plot after each fold.

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
    plt.figure(3)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='r', label='Random Chance', alpha=0.8)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('graphs', 'mean_ROC.png'))
    plt.clf()


def train_model_on_images(model, train_features, train_labels, num_epochs, val_features, val_labels):
    print("Training model")
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', \
    #        mode='min', min_delta = 0.05, patience = 20, restore_best_weights = True)
    history = model.fit(train_features, train_labels,
                        batch_size=BATCH_SIZE, epochs=num_epochs,
                        #        callbacks = [es_callback], \
                        validation_data=(val_features, val_labels),
                        verbose=2)
    return history


def create_confusion_matrix_and_roc_curve(model, val_features, val_labels, cm_file, tprs, mean_fpr, aucs):
    # Compute ROC curve and area the curve
    probas = model.predict_proba(val_features)[:, 1]  # 0 = definitely c1, 1 = definitely c2
    prob_classification = [round(a + 0.001) for a in probas]
    # Compute ROC curve and area the curve
    fpr, tpr, thresh = roc_curve(val_labels, probas)
    tn, fp, fn, tp = confusion_matrix(val_labels, prob_classification).ravel()
    confusion_mat = '\t\t   Predicted\n\t\t  P\t    N\t\n\t\t  --------------\n\t\tP|  ' + str(tp)
    confusion_mat += ' \t|  ' + str(fn)
    confusion_mat += ' \t|\nActual\t  --------------\n\t\tN|  ' + str(fp)
    confusion_mat += ' \t|  ' + str(tn)
    confusion_mat += ' \t|\n\t\t  --------------\n'
    print(confusion_mat)
    cm_file.write(confusion_mat)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # use the mean statistics to compare each model (that we train/test using 10-fold cv)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # plot the mean ROC curve and display AUC (mean/st dev)
    plot_roc_for_kfold(mean_fpr, mean_tpr, mean_auc, std_auc)
    return tn, fp, fn, tp


def save_results_to_csv(results):
    results = results.rename({0: 'Fold Number',
                              1: 'Training Loss',
                              2: 'Training Accuracy',
                              3: 'Validation Loss',
                              4: 'Validation Accuracy',
                              5: 'True Negatives', 6: 'False Positives',
                              7: 'False Negatives', 8: 'True Positives'})
    results.to_csv(os.path.join('graphs', 'final_acc_loss.csv'), encoding='utf-8', index=False)


def train_cross_validate(n_folds, features, labels, img_size, color, num_epochs):
    """ Import images from the file system and returns two numpy arrays containing the pixel information and classification.

    Parameters:
    -----
    @ n_folds : int
    Number of folds to train on (minimum of 2)

    @ features : np.array
    Array containing the pixel values -- dimensions are (img_size x img_size x 3) if color,
    (img_size x img_size x 1) if grayscale.

    @ labels : np.array
    Array containing the actual labels (0/1) of the images

    @ img_size : int
    Pixel dimensions of images

    @ num_epochs : int
    Maximum number of epochs to perform on each k-fold.

    Output:
    -----
    none
    """
    charts = ChartCreator()

    # initialize stratifying k fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # data frame to save values of loss and validation after each fold
    results = pd.DataFrame()

    # for roc plotting
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    cm_file = open(os.path.join('graphs', 'confusion_matrix.txt'), 'w')

    for index, (train_indices, val_indices) in enumerate(skf.split(features, labels)):
        print("Training on fold " + str(index + 1) + "/" + str(n_folds))
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        print("Training data obtained")
        val_features = features[val_indices]
        val_labels = labels[val_indices]
        print("Validation data obtained")

        # Create new model each time
        architecture = SmithsonianModel(img_size, color_mode=color, seed=SEED, lr=LEARNING_RATE)
        model = architecture.model
        history = train_model_on_images(model, train_features, train_labels,
                                        num_epochs, val_features, val_labels)

        model.save('saved_models/CNN_' + str(index + 1) + '.model')

        charts.plot_accuracy_and_loss(history, index)

        # Compute ROC curve and area the curve
        tn, fp, fn, tp = charts.create_confusion_matrix_and_roc_curve(model,
                                                                      val_features, val_labels, cm_file,
                                                                      tprs, mean_fpr, aucs)

        len_history = len(history.history['loss'])
        results = results.append([[index + 1,
                                   history.history['loss'][len_history - 1],
                                   history.history['acc'][len_history - 1],
                                   history.history['val_loss'][len_history - 1],
                                   history.history['val_acc'][len_history - 1],
                                   tn, fp, fn, tp]])

    cm_file.close()
    save_results_to_csv(results)


if __name__ == '__main__':
    start_time = time.time()

    # Set up
    n_folds, img_directory, folders, img_size, color, n_epochs = parse_arguments()
    create_folders()

    # Load in images and shuffle order
    images = ImageLoader(img_directory, folders, img_size, color, SEED)
    features = images.features
    labels = images.labels

    # Train model
    train_cross_validate(n_folds, features, labels, img_size, color, n_epochs)

    # end
    print('c1: ' + folders[0] + ', c2: ' + folders[1])
    end_time = time.time()
    print('Completed in %.1f seconds' % (end_time - start_time))
