import os
import random
import numpy as np
from pathlib import Path
import matplotlib
from labeled_images.labeledimages import LabeledImages
from models.smithsonian import SmithsonianModel
from models.model_training import ModelTrainer
from utilities.timer import Timer
from models.modeltrainingarguments import ModelTrainingArguments
import tensorflow as tf

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')

    cnn_arguments = ModelTrainingArguments()
    new_images = LabeledImages(SEED)

    new_images.load_training_images(cnn_arguments.training_image_folder, cnn_arguments.image_size,
                                    cnn_arguments.color_mode, shuffle=True, n_folds=cnn_arguments.n_folds)
    # multiclass architecture
    architecture = SmithsonianModel(SEED, cnn_arguments.lr, cnn_arguments.image_size, cnn_arguments.num_output_classes,
                                    cnn_arguments.color_mode)
    trainer = ModelTrainer(cnn_arguments.n_epochs, cnn_arguments.batch_size, cnn_arguments.n_folds, architecture, SEED)
    trainer.train_and_save_all_models(new_images)

    # generate charts
    trainer.charts.finalize()

    classes = (len(new_images.class_labels))
    for x in range(classes):
        print('class ' + str(x) + ': ' + new_images.class_labels[x])

    timer.stop()
    timer.print_results()


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    # delete previous ROC charts to generate new ones
    for fname in os.listdir('graphs'):
        if fname.startswith("mean_ROC"):
            os.remove(os.path.join('graphs', fname))

    main()
